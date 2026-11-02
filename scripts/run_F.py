"""Partie F : descelle le test. Usage : python scripts/run_F.py [Baseline E1 E2 D1 D2 C1 NLI similarite]
Avec --nyt : ajoute seulement les scores du test NYT aux résultats déjà calculés.
Les seuils viennent uniquement de val ; aucun choix de modèle n'est fait ici."""

import gc
import json
import re
import sys

import numpy as np
import pandas as pd
import torch

from valeval.data import RAW, VALUES, label_matrix, load_split
from valeval.evaluate import log_result, scores, seuil_unique, seuils_par_valeur
from valeval.finaux import CACHE, RACINE, C1, D1, D2, E1, E2, NLI, Baseline, Extracteur, ecrire_predictions, splits_finaux
from valeval.keywords import depuis_descriptions, extraire

SORTIES = CACHE / "final"
FAMILLES = {"Baseline": Baseline, "E1": E1, "E2": E2, "D1": D1, "D2": D2, "C1": C1, "NLI": NLI}
AVEC_F4 = {("Baseline", 0), ("E1", 42), ("C1", 42)}


def sans_mots(df, mots):
    """F4 : retire ces mots de la conclusion et de la prémisse ; la position ne change pas."""
    motif = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(mots))) + r")\b", re.IGNORECASE)
    res = df.copy()
    for col in ["conclusion", "premise"]:
        res[col] = [" ".join(motif.sub(" ", t).split()) or "." for t in res[col]]
    return res


def f4(m, test, y, p, s, dicos):
    """Parmi les bonnes détections (argument, valeur), combien survivent au retrait des mots-clés de cette valeur ?"""
    res = {}
    for nom_dico, dico in dicos.items():
        lignes = []
        for j, v in enumerate(VALUES):
            seuil = s[j] if np.ndim(s) else s
            vus = (p[:, j] >= seuil) & (y[:, j] == 1)
            if vus.sum() == 0:
                continue
            p2 = m.probas(sans_mots(test[vus], dico[v]))[:, j]
            lignes.append({"valeur": v, "detectes": int(vus.sum()), "restent": int((p2 >= seuil).sum())})
        res[nom_dico] = lignes
    return res


def traiter(cle, graine, splits, val, y_va, tr, y_tr, dicos):
    famille = FAMILLES[cle]
    m = famille(tr, y_tr, val, y_va, graine)
    # les scores NLI de val existent déjà (Partie B1) : inutile de les recalculer
    p_va = np.load(CACHE / "zs_deberta_large_en_validation.npy") if cle == "NLI" else m.probas(val)
    if getattr(famille, "seuil_par_valeur", False):
        s = seuils_par_valeur(y_va, p_va, quantiles=True)
    else:
        s = seuil_unique(y_va, p_va)
    nyt = list(splits) == ["test-nyt"]
    res = {"seuil_nyt" if nyt else "seuil": np.asarray(s).tolist()}
    if not nyt:
        np.save(SORTIES / f"{famille.nom}_seed{graine}_validation.npy", p_va)
    for nom, df in splits.items():
        p = m.probas(df)
        np.save(SORTIES / f"{famille.nom}_seed{graine}_{nom}.npy", p)
        res[nom] = scores(label_matrix(df), p >= s)
        if nom == "test":
            ecrire_predictions(RACINE / "predictions" / f"{famille.nom}_seed{graine}.csv", df, p, p >= s, VALUES)
            log_result(famille.nom, "final", graine, val=scores(y_va, p_va >= s), test=res[nom], notes="Partie F")
            if (cle, graine) in AVEC_F4:
                res["f4"] = f4(m, df, label_matrix(df), p, s, dicos)
    chemin = SORTIES / f"{famille.nom}_seed{graine}.json"
    res = {**(json.loads(chemin.read_text(encoding="utf-8")) if nyt else {}), **res}
    chemin.write_text(json.dumps(res, default=float), encoding="utf-8")
    print(famille.nom, graine, {k: round(v["macro_f1"], 3) for k, v in res.items() if isinstance(v, dict) and "macro_f1" in v}, flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()


def similarite():
    """F3 : pour chaque conclusion du test, similarité avec la conclusion du train la plus proche."""
    ext = Extracteur()
    def vecteurs(textes):
        v = Extracteur.moyenne(*ext.etats(list(textes)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    tr, te = load_split("training"), load_split("test")
    c_tr, c_te = sorted(set(tr.conclusion)), te.conclusion.tolist()
    sim = vecteurs(c_te) @ vecteurs(c_tr).T
    np.save(SORTIES / "similarite_test.npy", sim.max(1))
    pd.DataFrame({"conclusion": c_te, "voisin_train": [c_tr[i] for i in sim.argmax(1)],
                  "similarite": sim.max(1).round(3)}).to_json(SORTIES / "voisins_test.json", force_ascii=False)
    print("conclusions du test exactement dans le train :", len(set(c_te) & set(c_tr)), flush=True)


def main():
    SORTIES.mkdir(parents=True, exist_ok=True)
    (RACINE / "predictions").mkdir(exist_ok=True)
    tr, val = load_split("training"), load_split("validation")
    y_tr, y_va = label_matrix(tr), label_matrix(val)
    demandes = sys.argv[1:]
    splits = splits_finaux()
    if "--nyt" in demandes:
        demandes.remove("--nyt")
        splits = {"test-nyt": splits["test-nyt"]}
    print("splits :", {k: len(v) for k, v in splits.items()}, flush=True)
    dicos = {"train_k200": extraire(tr, 200), "descriptions": depuis_descriptions(RAW / "value-categories.json")}
    for cle in demandes or list(FAMILLES) + ["similarite"]:
        if cle == "similarite":
            similarite()
            continue
        cle, _, seule = cle.partition("@")
        for graine in [int(seule)] if seule else FAMILLES[cle].graines:
            traiter(cle, graine, splits, val, y_va, tr, y_tr, dicos)


if __name__ == "__main__":
    main()
