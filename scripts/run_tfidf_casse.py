"""Q3 : effet de la casse, mesuré sur la baseline (GloVe étant en minuscules, C1 ne peut pas le tester)."""

import json
from pathlib import Path

from valeval.baseline import TfidfLR, texte_complet
from valeval.data import label_matrix, load_split
from valeval.evaluate import log_result, scores, seuil_unique

tr, va = load_split("training"), load_split("validation")
y_tr, y_va = label_matrix(tr), label_matrix(va)
res = {}
for nom, minuscules in [("minuscules", True), ("casse conservée", False)]:
    p = TfidfLR(C=1, lowercase=minuscules).fit(texte_complet(tr), y_tr).predict_proba(texte_complet(va))
    s = seuil_unique(y_va, p)
    res[nom] = scores(y_va, p >= s)["macro_f1"]
    log_result("baseline_tfidf_lr", nom, 0, val=scores(y_va, p >= s), notes=f"ablation casse, seuil {s:.2f}")
(Path(__file__).resolve().parents[1] / "cache" / "runs" / "tfidf_casse.json").write_text(json.dumps(res), encoding="utf-8")
print(res)
