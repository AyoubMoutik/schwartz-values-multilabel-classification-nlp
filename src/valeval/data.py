"""Chargement des splits Touché23-ValueEval (textes + labels)."""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Les 20 catégories de valeurs, dans l'ordre des colonnes des fichiers labels-*.tsv
VALUES = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    "Achievement", "Power: dominance", "Power: resources", "Face",
    "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
    "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance", "Universalism: objectivity",
]

SPLITS = ["training", "validation", "test", "test-nahjalbalagha", "validation-zhihu"]


def load_split(split: str) -> pd.DataFrame:
    """Un argument par ligne : id, conclusion, stance, premise + 20 colonnes 0/1."""
    args = pd.read_csv(RAW / f"arguments-{split}.tsv", sep="\t")
    labels = pd.read_csv(RAW / f"labels-{split}.tsv", sep="\t")
    df = args.merge(labels, on="Argument ID", how="inner", validate="one_to_one")
    assert len(df) == len(args) == len(labels), f"{split}: textes et labels ne correspondent pas"
    assert list(labels.columns[1:]) == VALUES, f"{split}: colonnes de labels inattendues"
    df = df.rename(columns={"Argument ID": "id", "Conclusion": "conclusion",
                            "Stance": "stance", "Premise": "premise"})
    return df


def label_matrix(df: pd.DataFrame):
    """Matrice (n_arguments, 20) de 0/1 : la cible y des modèles."""
    return df[VALUES].to_numpy()
