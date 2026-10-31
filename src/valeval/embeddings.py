"""Partie C : vecteurs GloVe pré-entraînés (glove.6B.300d)."""

from pathlib import Path

import numpy as np

GLOVE = Path(__file__).resolve().parents[2] / "cache" / "glove" / "glove.6B.300d.txt"


def charger_glove(chemin=GLOVE) -> tuple:
    """Renvoie (liste des mots, matrice (n_mots, 300))."""
    mots, vecteurs = [], []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            mot, *nombres = ligne.rstrip("\n").split(" ")
            mots.append(mot)
            vecteurs.append(np.array(nombres, dtype=np.float32))
    return mots, np.stack(vecteurs)


def voisins(requete: np.ndarray, mots: list, matrice: np.ndarray, n: int = 6, exclure=()) -> list:
    """Les n mots dont le vecteur est le plus proche de requete (similarité cosinus)."""
    norme = matrice / np.linalg.norm(matrice, axis=1, keepdims=True)
    cos = norme @ (requete / np.linalg.norm(requete))
    res = []
    for i in np.argsort(-cos):
        if mots[i] not in exclure:
            res.append((mots[i], round(float(cos[i]), 2)))
        if len(res) == n:
            return res


def etendre_avec_glove(vocab, g_mots: list, g_mat: np.ndarray, graine: int = 42) -> np.ndarray:
    """Ajoute au vocabulaire du train les mots de GloVe et renvoie la matrice d'embeddings.
    <pad> reste à 0 ; les mots sans vecteur GloVe (<unk>, <sep>, fautes de frappe) reçoivent un vecteur aléatoire."""
    g_idx = {m: i for i, m in enumerate(g_mots)}
    for m in g_mots:
        if m not in vocab.indice:
            vocab.indice[m] = len(vocab.mots)
            vocab.mots.append(m)
    rng = np.random.default_rng(graine)
    matrice = rng.normal(0, g_mat.std(), (len(vocab), g_mat.shape[1])).astype(np.float32)
    matrice[0] = 0
    for i, m in enumerate(vocab.mots):
        if m in g_idx:
            matrice[i] = g_mat[g_idx[m]]
    return matrice
