"""Partie E. Usage : python scripts/run_E.py roberta@42 deberta@43 roberta@42:0.5 roberta@42~8 (modèle@graine:lambda~époques)"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from valeval.data import label_matrix, load_split
from valeval.evaluate import log_result, scores, seuil_unique
from valeval.plm import entrainer

RACINE = Path(__file__).resolve().parents[1]
SORTIES, POIDS = RACINE / "cache" / "runs", RACINE / "cache" / "ckpt"


def executer(modele, graine, lam, epoques):
    tr, va = load_split("training"), load_split("validation")
    y_tr, y_va = label_matrix(tr), label_matrix(va)
    nom = f"E2_{modele}_lam{lam}" if lam > 0 else f"E1_{modele}"
    nom += f"_ep{epoques}" if epoques != 5 else ""
    print(nom, "graine", graine, flush=True)
    res = entrainer(modele, tr, va, y_tr, y_va, graine, lam=lam, epoques=epoques)

    p, etat = res.pop("probas_val"), res.pop("etat")
    s = seuil_unique(y_va, p)
    val = scores(y_va, p >= s)
    log_result(nom, "texte brut", graine, val=val, temps_s=round(res["temps_s"], 1), params=res["params"],
               notes=f"seuil {s:.2f}, {len(res['courbes'])} époques")
    SORTIES.mkdir(parents=True, exist_ok=True)
    POIDS.mkdir(parents=True, exist_ok=True)
    np.save(SORTIES / f"{nom}_seed{graine}_val.npy", p)
    torch.save(etat, POIDS / f"{nom}_seed{graine}.pt")
    (SORTIES / f"{nom}_seed{graine}.json").write_text(
        json.dumps({**res, "seuil": s, "macro_f1": val["macro_f1"]}), encoding="utf-8")
    print(f"{nom} graine {graine}: macro-F1 val {val['macro_f1']:.3f} (seuil {s:.2f}), {res['temps_s']:.0f}s, "
          f"{res['params']:,} paramètres", flush=True)


if __name__ == "__main__":
    for demande in sys.argv[1:]:
        m = re.fullmatch(r"(\w+)(?:@(\d+))?(?::([\d.]+))?(?:~(\d+))?", demande)
        executer(m[1], int(m[2] or 42), float(m[3] or 0), int(m[4] or 5))
        torch.cuda.empty_cache()
