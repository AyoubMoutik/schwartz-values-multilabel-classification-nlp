"""B1 : scores NLI sur val pour chaque modèle et formulation. Résultats dans cache/."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

from valeval.baseline import texte_complet
from valeval.data import load_split
from valeval.hypotheses import FORMULATIONS, liste
from valeval.zeroshot import MODELES, charger, score_implication

CACHE = Path(__file__).resolve().parents[1] / "cache"
BATCH = {"mdeberta": 128, "bart": 96, "deberta_large": 64}


def main(split="validation"):
    textes = texte_complet(load_split(split)).tolist()
    for nom in MODELES:
        tok, modele = charger(nom)
        for formulation in FORMULATIONS:
            sortie = CACHE / f"zs_{nom}_{formulation}_{split}.npy"
            if sortie.exists():
                continue
            debut = time.time()
            p = score_implication(tok, modele, textes, liste(formulation), BATCH[nom])
            np.save(sortie, p)
            print(f"{nom} {formulation}: {time.time() - debut:.0f}s", flush=True)
        del modele
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(*sys.argv[1:])
