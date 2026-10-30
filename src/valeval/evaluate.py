"""Métriques multi-label et journal results/results.csv."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from valeval.data import VALUES

RESULTS = Path(__file__).resolve().parents[2] / "results" / "results.csv"
COLUMNS = ["date", "modele", "pretraitement", "graine", "macro_f1_val", "macro_f1_test",
           "micro_f1_val", "micro_f1_test", "temps_entrainement_s", "params_entraines", "notes"]


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Scores d'un ensemble de prédictions 0/1 de forme (n_arguments, 20)."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy_cases": (y_true == y_pred).mean(),      # part des 20×n cases correctes
        "accuracy_exacte": accuracy_score(y_true, y_pred),  # part d'arguments 100 % corrects
    }


def f1_par_valeur(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(zip(VALUES, f1_score(y_true, y_pred, average=None, zero_division=0)))


SEUILS = np.arange(0.05, 0.96, 0.01)


def grille_quantiles(scores_val: np.ndarray) -> np.ndarray:
    """Seuils suivant la répartition des scores : utile quand ils sont écrasés près de 0 ou de 1."""
    return np.unique(np.quantile(scores_val, np.linspace(0.01, 0.99, 99)))


def seuil_unique(y_true: np.ndarray, probas: np.ndarray, grille=SEUILS) -> float:
    """Le seuil, commun aux 20 valeurs, qui maximise le macro-F1."""
    f1s = [f1_score(y_true, probas >= s, average="macro", zero_division=0) for s in grille]
    return float(grille[int(np.argmax(f1s))])


def seuils_par_valeur(y_true: np.ndarray, probas: np.ndarray, quantiles: bool = False) -> np.ndarray:
    """Un seuil par valeur : chacun maximise le F1 de sa valeur."""
    res = np.zeros(y_true.shape[1])
    for j in range(y_true.shape[1]):
        grille = grille_quantiles(probas[:, j]) if quantiles else SEUILS
        f1s = [f1_score(y_true[:, j], probas[:, j] >= s, zero_division=0) for s in grille]
        res[j] = grille[int(np.argmax(f1s))]
    return res


def log_result(modele, pretraitement, graine, val=None, test=None,
               temps_s=None, params=None, notes=""):
    """Ajoute une ligne au journal. val et test sont des dict issus de scores()."""
    RESULTS.parent.mkdir(exist_ok=True)
    nouveau = not RESULTS.exists()
    val, test = val or {}, test or {}
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nouveau:
            w.writerow(COLUMNS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), modele, pretraitement, graine,
                    val.get("macro_f1"), test.get("macro_f1"), val.get("micro_f1"), test.get("micro_f1"),
                    temps_s, params, notes])
