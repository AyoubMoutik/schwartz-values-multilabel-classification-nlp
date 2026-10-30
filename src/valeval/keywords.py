"""Dictionnaire de mots-clés par valeur, extrait du train uniquement."""

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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


def depuis_descriptions(chemin, max_valeurs: int = 2) -> dict:
    """Dictionnaire sans aucun exemple annoté : les mots de value-categories.json.
    On retire les mots vides et ceux partagés par plus de max_valeurs valeurs (allowing, being…)."""
    desc = json.loads(chemin.read_text(encoding="utf-8"))
    bruts = {v: set() for v in VALUES}
    for v in VALUES:
        for nom, exemples in desc[v].items():
            bruts[v] |= mots(nom + " " + " ".join(exemples))
    usage = pd.Series([m for ens in bruts.values() for m in ens]).value_counts()
    garde = lambda m: len(m) >= 4 and m not in ENGLISH_STOP_WORDS and usage[m] <= max_valeurs
    return {v: sorted(filter(garde, bruts[v])) for v in VALUES}
