"""Dictionnaire de mots-clés par valeur, extrait du train uniquement."""

import re

import numpy as np
import pandas as pd

from valeval.data import VALUES

MOT = re.compile(r"[a-z']+")


def mots(texte: str) -> set:
    return set(MOT.findall(texte.lower()))


def extraire(df: pd.DataFrame, k: int, min_docs: int = 5) -> dict:
    """Pour chaque valeur, les k mots les plus typiques de ses arguments (log-odds lissé)."""
    docs = df.premise.map(mots)
    tous = pd.Series([m for d in docs for m in d]).value_counts()
    vocab = tous[tous >= min_docs].index
    dico = {}
    for v in VALUES:
        pos = df[v] == 1
        n_pos = pd.Series([m for d in docs[pos] for m in d]).value_counts().reindex(vocab, fill_value=0)
        n_neg = tous.reindex(vocab) - n_pos
        # log(chance du mot dans les positifs / chance dans les négatifs) ; +1 évite les divisions par 0
        odds = np.log((n_pos + 1) / (pos.sum() - n_pos + 1)) - np.log((n_neg + 1) / ((~pos).sum() - n_neg + 1))
        dico[v] = list(odds.sort_values(ascending=False).index[:k])
    return dico


def predire(df: pd.DataFrame, dico: dict) -> np.ndarray:
    """1 si le texte contient au moins un mot-clé de la valeur."""
    docs = (df.conclusion + " " + df.premise).map(mots)
    return np.array([[int(bool(d & set(dico[v]))) for v in VALUES] for d in docs])
