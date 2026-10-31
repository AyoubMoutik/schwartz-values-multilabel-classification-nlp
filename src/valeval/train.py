"""Boucle d'entraînement commune aux réseaux de la Partie C."""

import copy
import random
import time

import numpy as np
import torch
from torch import nn

from valeval.evaluate import scores, seuil_unique

GRILLE_RAPIDE = np.arange(0.10, 0.90, 0.02)


def fixer_graines(graine: int) -> None:
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def lots(ids: list, y: np.ndarray, taille: int, melanger: bool, tirage=None):
    """Découpe en lots ; chaque lot est complété par des 0 jusqu'à son texte le plus long."""
    ordre = np.arange(len(ids)) if tirage is None else tirage
    if melanger:
        ordre = np.random.permutation(ordre)
    for i in range(0, len(ordre), taille):
        choix = ordre[i:i + taille]
        n = max(len(ids[j]) for j in choix)
        x = torch.zeros(len(choix), n, dtype=torch.long)
        for ligne, j in enumerate(choix):
            x[ligne, :len(ids[j])] = torch.as_tensor(ids[j])
        yield x.cuda(), x.cuda() != 0, torch.as_tensor(y[choix], dtype=torch.float32).cuda()


@torch.inference_mode()
def predire_probas(modele, ids: list, y: np.ndarray, taille: int = 256) -> np.ndarray:
    modele.eval()
    return np.concatenate([modele(x, m).sigmoid().float().cpu().numpy() for x, m, _ in lots(ids, y, taille, False)])


def poids_positifs(y_tr: np.ndarray) -> torch.Tensor:
    """Perte pondérée : une erreur sur un positif rare coûte (négatifs / positifs) fois plus."""
    freq = y_tr.mean(0)
    return torch.as_tensor((1 - freq) / freq, dtype=torch.float32).cuda()


def tirage_surechantillonne(y_tr: np.ndarray, graine: int) -> np.ndarray:
    """Sur-échantillonnage : un argument est tiré selon la rareté de sa valeur la plus rare."""
    rarete = 1 / y_tr.mean(0)
    poids = (y_tr * rarete).max(axis=1) + 1e-6
    return np.random.default_rng(graine).choice(len(y_tr), len(y_tr), replace=True, p=poids / poids.sum())


def nb_parametres(modele) -> int:
    return sum(p.numel() for p in modele.parameters() if p.requires_grad)


def entrainer(modele, ids_tr, y_tr, ids_va, y_va, epoques=30, patience=5, lr=1e-3,
              taille=32, pos_weight=None, tirage=None) -> dict:
    """Entraîne, garde l'époque au meilleur macro-F1 sur val, s'arrête après patience époques sans progrès."""
    modele.cuda()
    perte_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiseur = torch.optim.AdamW([p for p in modele.parameters() if p.requires_grad], lr=lr)
    meilleur, etat, sans_progres, courbes = -1.0, None, 0, []
    debut = time.time()
    for epoque in range(1, epoques + 1):
        modele.train()
        total, n = 0.0, 0
        for x, m, y in lots(ids_tr, y_tr, taille, True, tirage):
            optimiseur.zero_grad()
            perte = perte_fn(modele(x, m), y)      # 1. erreur sur ce lot
            perte.backward()                       # 2. rôle de chaque paramètre dans l'erreur
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optimiseur.step()                      # 3. petite correction des paramètres
            total, n = total + perte.item() * len(y), n + len(y)
        p_va = predire_probas(modele, ids_va, y_va)
        perte_va = float(perte_fn(torch.as_tensor(np.log(np.clip(p_va, 1e-6, 1 - 1e-6) / (1 - np.clip(p_va, 1e-6, 1 - 1e-6)))).cuda(),
                                  torch.as_tensor(y_va, dtype=torch.float32).cuda()))
        s = seuil_unique(y_va, p_va, GRILLE_RAPIDE)
        f1 = scores(y_va, p_va >= s)["macro_f1"]
        courbes.append({"epoque": epoque, "perte_train": total / n, "perte_val": perte_va, "macro_f1_val": f1})
        if f1 > meilleur:
            meilleur, etat, sans_progres = f1, copy.deepcopy(modele.state_dict()), 0
        else:
            sans_progres += 1
            if sans_progres >= patience:
                break
    modele.load_state_dict(etat)
    return {"courbes": courbes, "macro_f1_val": meilleur, "temps_s": time.time() - debut,
            "params": nb_parametres(modele), "probas_val": predire_probas(modele, ids_va, y_va)}
