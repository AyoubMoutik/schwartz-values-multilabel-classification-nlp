"""Partie E : fine-tuning d'un PLM, mono-tâche (E1) ou multi-tâche (E2)."""

import copy
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from valeval.data import VALUES
from valeval.evaluate import scores, seuil_unique
from valeval.train import GRILLE_RAPIDE, fixer_graines, nb_parametres, poids_positifs

MODELES = {"roberta": "roberta-base", "deberta": "microsoft/deberta-v3-base"}
MAX_LEN = 128

# Dimensions de Schwartz : Hedonism, Face et Humility se situent à la frontière de deux dimensions
DIMENSIONS = {
    "Ouverture au changement": ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism"],
    "Affirmation de soi": ["Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face"],
    "Continuité": ["Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
                   "Conformity: interpersonal", "Humility"],
    "Dépassement de soi": ["Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern",
                           "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity"],
}


def cibles_dimensions(y: np.ndarray) -> np.ndarray:
    """Une dimension vaut 1 si au moins une de ses valeurs vaut 1 ; forme (n, 4)."""
    return np.stack([y[:, [VALUES.index(v) for v in vs]].max(axis=1) for vs in DIMENSIONS.values()], axis=1)


class PLM(nn.Module):
    """Encodeur pré-entraîné + tête 20 valeurs, et tête 4 dimensions si multi-tâche."""

    def __init__(self, nom: str, multitache: bool = False):
        super().__init__()
        self.enc = AutoModel.from_pretrained(nom, dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.valeurs = nn.Linear(h, 20)
        self.dimensions = nn.Linear(h, 4) if multitache else None

    def forward(self, input_ids, attention_mask):
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.valeurs(cls), (self.dimensions(cls) if self.dimensions is not None else None)


def entrees(df, tok) -> list:
    """conclusion [SEP] position [SEP] prémisse, texte brut."""
    textes = (df.conclusion + tok.sep_token + df.stance + tok.sep_token + df.premise).tolist()
    return tok(textes, truncation=True, max_length=MAX_LEN).input_ids


def lots(ids, cibles, taille, melanger, pad):
    ordre = np.random.permutation(len(ids)) if melanger else np.arange(len(ids))
    for i in range(0, len(ordre), taille):
        choix = ordre[i:i + taille]
        n = max(len(ids[j]) for j in choix)
        x = torch.full((len(choix), n), pad, dtype=torch.long)
        for ligne, j in enumerate(choix):
            x[ligne, :len(ids[j])] = torch.as_tensor(ids[j])
        yield x.cuda(), (x != pad).cuda(), [torch.as_tensor(c[choix], dtype=torch.float32).cuda() for c in cibles]


@torch.inference_mode()
def probas(modele, ids, pad, taille=128) -> np.ndarray:
    modele.eval()
    res = []
    for x, m, _ in lots(ids, [np.zeros((len(ids), 1))], taille, False, pad):
        with torch.autocast("cuda", dtype=torch.float16):
            res.append(modele(x, m)[0].sigmoid().float().cpu().numpy())
    return np.concatenate(res)


def entrainer(nom, tr, va, y_tr, y_va, graine, lam=0.0, epoques=5, patience=2, lr=2e-5, taille=32):
    """lr 2e-5, batch 32, warmup 10 %, weight decay 0,01, fp16, arrêt anticipé sur le macro-F1 de val."""
    fixer_graines(graine)
    tok = AutoTokenizer.from_pretrained(MODELES[nom])
    ids_tr, ids_va = entrees(tr, tok), entrees(va, tok)
    pad = tok.pad_token_id
    modele = PLM(MODELES[nom], multitache=lam > 0).cuda()

    d_tr = cibles_dimensions(y_tr)
    perte_v = nn.BCEWithLogitsLoss(pos_weight=poids_positifs(y_tr))
    perte_d = nn.BCEWithLogitsLoss(pos_weight=poids_positifs(d_tr))
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=0.01)
    pas = epoques * int(np.ceil(len(ids_tr) / taille))
    planning = get_linear_schedule_with_warmup(optimiseur, int(0.1 * pas), pas)
    echelle = torch.amp.GradScaler("cuda")

    meilleur, etat, sans_progres, courbes = -1.0, None, 0, []
    debut = time.time()
    for epoque in range(1, epoques + 1):
        modele.train()
        total, n = 0.0, 0
        for x, m, (y, d) in lots(ids_tr, [y_tr, d_tr], taille, True, pad):
            optimiseur.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                sortie_v, sortie_d = modele(x, m)
                perte = perte_v(sortie_v, y)
                if lam > 0:
                    perte = perte + lam * perte_d(sortie_d, d)
            echelle.scale(perte).backward()
            echelle.unscale_(optimiseur)
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            echelle.step(optimiseur)
            echelle.update()
            planning.step()
            total, n = total + perte.item() * len(y), n + len(y)
        p = probas(modele, ids_va, pad)
        s = seuil_unique(y_va, p, GRILLE_RAPIDE)
        f1 = scores(y_va, p >= s)["macro_f1"]
        courbes.append({"epoque": epoque, "perte_train": total / n, "macro_f1_val": f1, "temps_s": time.time() - debut})
        print(f"  époque {epoque}: perte train {total / n:.3f}, macro-F1 val {f1:.3f}, {time.time() - debut:.0f}s", flush=True)
        if f1 > meilleur:
            meilleur, sans_progres = f1, 0
            etat = copy.deepcopy({k: v.cpu() for k, v in modele.state_dict().items()})
        else:
            sans_progres += 1
            if sans_progres >= patience:
                break
    modele.load_state_dict(etat)
    return {"courbes": courbes, "macro_f1_val": meilleur, "temps_s": time.time() - debut,
            "params": nb_parametres(modele), "probas_val": probas(modele, ids_va, pad), "etat": etat}
