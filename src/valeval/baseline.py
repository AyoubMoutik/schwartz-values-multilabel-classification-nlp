"""Baseline A : TF-IDF (mots 1-2 + caractères 2-5) + régression logistique one-vs-rest."""

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def texte_complet(df: pd.DataFrame) -> pd.Series:
    """Conclusion, position et prémisse concaténées en une seule chaîne."""
    return df.conclusion + " . " + df.stance + " . " + df.premise


class TfidfLR:
    def __init__(self, C=1.0, lowercase=True):
        self.mots = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=lowercase)
        self.cars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, lowercase=lowercase)
        self.clf = OneVsRestClassifier(
            LogisticRegression(C=C, class_weight="balanced", max_iter=1000), n_jobs=-1)

    def fit(self, textes, y):
        # le vocabulaire et les poids IDF sont appris sur le train uniquement
        X = hstack([self.mots.fit_transform(textes), self.cars.fit_transform(textes)]).tocsr()
        self.clf.fit(X, y)
        return self

    def predict_proba(self, textes):
        X = hstack([self.mots.transform(textes), self.cars.transform(textes)]).tocsr()
        return self.clf.predict_proba(X)
