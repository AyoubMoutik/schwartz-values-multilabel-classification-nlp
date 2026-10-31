"""Partie D : roberta-base gelé. Usage : python scripts/run_D.py extraire d1 d2@42 d2@43 d2@44"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from valeval.data import label_matrix, load_split
from valeval.evaluate import log_result, scores, seuil_unique
from valeval.models import BiLSTMAttention, Etats
from valeval.train import entrainer, fixer_graines, lots_etats

RACINE = Path(__file__).resolve().parents[1]
CACHE, SORTIES = RACINE / "cache" / "roberta", RACINE / "cache" / "runs"
NOM, MAX_LEN = "roberta-base", 128


def entree(df, sep):
    """conclusion [SEP] position [SEP] prémisse, texte brut."""
    return (df.conclusion + sep + df.stance + sep + df.premise).tolist()


@torch.inference_mode()
def extraire():
    CACHE.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NOM)
    modele = AutoModel.from_pretrained(NOM, dtype=torch.float32).cuda().eval()
    for split in ["training", "validation"]:
        textes = entree(load_split(split), tok.sep_token)
        etats, masques = [], []
        debut = time.time()
        for i in range(0, len(textes), 64):
            x = tok(textes[i:i + 64], truncation=True, max_length=MAX_LEN, padding="max_length",
                    return_tensors="pt").to("cuda")
            with torch.autocast("cuda"):
                sortie = modele(**x).last_hidden_state
            etats.append(sortie.half().cpu().numpy())
            masques.append(x.attention_mask.bool().cpu().numpy())
        np.save(CACHE / f"etats_{split}.npy", np.concatenate(etats))
        np.save(CACHE / f"masques_{split}.npy", np.concatenate(masques))
        print(f"{split}: {time.time() - debut:.0f}s", flush=True)


def charger(split):
    return np.load(CACHE / f"etats_{split}.npy", mmap_mode="r"), np.load(CACHE / f"masques_{split}.npy")


def resumes(etats, masques):
    """Deux résumés par argument : le token [CLS], et la moyenne des tokens pondérée par le masque."""
    cls = np.asarray(etats[:, 0], dtype=np.float32)
    moyenne = np.zeros_like(cls)
    for i in range(0, len(cls), 512):
        e, m = np.asarray(etats[i:i + 512], dtype=np.float32), masques[i:i + 512, :, None]
        moyenne[i:i + 512] = (e * m).sum(1) / m.sum(1)
    return {"CLS": cls, "moyenne": moyenne}


def d1():
    y_tr, y_va = label_matrix(load_split("training")), label_matrix(load_split("validation"))
    r_tr, r_va = resumes(*charger("training")), resumes(*charger("validation"))
    lignes = []
    for pooling in r_tr:
        echelle = StandardScaler().fit(r_tr[pooling])      # ajustée sur le train uniquement
        x_tr, x_va = echelle.transform(r_tr[pooling]), echelle.transform(r_va[pooling])
        for C in [0.00003, 0.0001, 0.0003, 0.001, 0.01, 0.1, 1]:
            debut = time.time()
            clf = OneVsRestClassifier(LogisticRegression(C=C, class_weight="balanced", max_iter=2000)).fit(x_tr, y_tr)
            p = clf.predict_proba(x_va)
            s = seuil_unique(y_va, p)
            val = scores(y_va, p >= s)
            duree = time.time() - debut
            lignes.append({"pooling": pooling, "C": C, "seuil": s, "macro_f1": val["macro_f1"], "temps_s": duree})
            log_result(f"D1_roberta_gele_{pooling}", "texte brut", 0, val=val, temps_s=round(duree, 1),
                       params=20 * 769, notes=f"C={C}, seuil {s:.2f}")
            print(lignes[-1], flush=True)
    (SORTIES / "D1.json").write_text(json.dumps(lignes), encoding="utf-8")


@torch.inference_mode()
def sauver_attention(modele, donnees_va, y_va, graine):
    tok = AutoTokenizer.from_pretrained(NOM)
    modele.eval()
    alphas = []
    for x, m, _ in lots_etats(donnees_va, y_va, 256, False):
        alphas += [a[:int(n)].tolist() for a, n in zip(modele.poids(x, m).cpu().numpy(), m.sum(1).cpu())]
    textes = entree(load_split("validation"), tok.sep_token)
    tokens = [tok.convert_ids_to_tokens(i) for i in tok(textes, truncation=True, max_length=MAX_LEN).input_ids]
    (SORTIES / f"D2_seed{graine}_attention.json").write_text(
        json.dumps([{"tokens": t, "alpha": a} for t, a in zip(tokens, alphas)]), encoding="utf-8")


def d2(graine):
    y_tr = label_matrix(load_split("training"))
    y_va = label_matrix(load_split("validation"))
    e_tr, m_tr = charger("training")
    e_va, m_va = charger("validation")
    donnees_tr = (torch.as_tensor(np.array(e_tr)), torch.as_tensor(m_tr))
    donnees_va = (torch.as_tensor(np.array(e_va)), torch.as_tensor(m_va))
    fixer_graines(graine)
    modele = BiLSTMAttention(Etats(768))
    res = entrainer(modele, donnees_tr, y_tr, donnees_va, y_va, fabrique=lots_etats)
    p = res.pop("probas_val")
    s = seuil_unique(y_va, p)
    val = scores(y_va, p >= s)
    log_result("D2_roberta_gele_bilstm_attention", "texte brut", graine, val=val, temps_s=round(res["temps_s"], 1),
               params=res["params"], notes=f"seuil {s:.2f}, {len(res['courbes'])} époques")
    np.save(SORTIES / f"D2_seed{graine}_val.npy", p)
    (SORTIES / f"D2_seed{graine}.json").write_text(
        json.dumps({**res, "seuil": s, "macro_f1": val["macro_f1"]}), encoding="utf-8")
    sauver_attention(modele, donnees_va, y_va, graine)
    print(f"D2 graine {graine}: macro-F1 val {val['macro_f1']:.3f} (seuil {s:.2f}), {len(res['courbes'])} époques, "
          f"{res['temps_s']:.0f}s, {res['params']:,} paramètres", flush=True)


if __name__ == "__main__":
    for demande in sys.argv[1:]:
        tache, _, graine = demande.partition("@")
        if tache == "d2":
            d2(int(graine or 42))
        else:
            {"extraire": extraire, "d1": d1}[tache]()
