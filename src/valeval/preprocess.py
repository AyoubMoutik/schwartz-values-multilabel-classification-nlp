"""Partie C : tokenisation, vocabulaire (train uniquement) et encodage en indices."""

import re
from collections import Counter

import pandas as pd

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
# Contractions coupées comme dans GloVe (can't -> ca + n't, child's -> child + 's) : la négation reste un token
TOKEN = re.compile(r"[a-z0-9]+(?=n't)|n't|'(?:s|re|ve|ll|d|m)\b|[a-z0-9]+|[.,!?;:]")


def tokeniser(texte: str) -> list:
    return TOKEN.findall(texte.lower())


def tokens_argument(ligne) -> list:
    """Conclusion <sep> position <sep> prémisse."""
    return (tokeniser(ligne.conclusion) + [SEP] + tokeniser(ligne.stance) + [SEP]
            + tokeniser(ligne.premise))


def corpus(df: pd.DataFrame) -> list:
    return [tokens_argument(l) for l in df.itertuples()]


class Vocabulaire:
    def __init__(self, textes: list, min_freq: int = 2):
        """textes : listes de tokens du train uniquement."""
        self.freq = Counter(t for texte in textes for t in texte if t != SEP)
        gardes = sorted(t for t, n in self.freq.items() if n >= min_freq)
        self.mots = [PAD, UNK, SEP] + gardes
        self.indice = {m: i for i, m in enumerate(self.mots)}

    def __len__(self):
        return len(self.mots)

    def encoder(self, tokens: list, max_len: int) -> list:
        unk = self.indice[UNK]
        return [self.indice.get(t, unk) for t in tokens[:max_len]]


def texte_caracteres(ligne) -> str:
    return f"{ligne.conclusion} | {ligne.stance} | {ligne.premise}".lower()


class VocabulaireCaracteres:
    def __init__(self, textes: list, min_freq: int = 2):
        freq = Counter(c for texte in textes for c in texte)
        self.indice = {c: i + 2 for i, c in enumerate(sorted(c for c, n in freq.items() if n >= min_freq))}

    def __len__(self):
        return len(self.indice) + 2       # 0 = remplissage, 1 = caractère inconnu

    def encoder(self, texte: str, max_len: int) -> list:
        return [self.indice.get(c, 1) for c in texte[:max_len]]


NEGATIONS = {"not", "n't", "no", "never", "nor", "neither", "none", "cannot"}
PONCTUATION = {".", ",", "!", "?", ";", ":"}


def retirer(textes: list, mots: set) -> list:
    """Ablation : supprime ces tokens ; les séparateurs <sep> restent."""
    return [[t for t in texte if t not in mots] for texte in textes]
