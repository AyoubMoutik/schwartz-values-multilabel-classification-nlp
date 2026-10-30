"""Partie B1 : classification zero-shot par NLI, sans aucun entraînement."""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELES = {
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "bart": "facebook/bart-large-mnli",
    # ajouté par nous : modèle anglais conçu pour le zero-shot (classes : entailment / not_entailment)
    "deberta_large": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
}


def charger(nom: str):
    tok = AutoTokenizer.from_pretrained(MODELES[nom])
    modele = AutoModelForSequenceClassification.from_pretrained(MODELES[nom], dtype=torch.float32).cuda().eval()
    return tok, modele


def classes(modele) -> dict:
    """Numéro de chaque sortie : il change d'un modèle à l'autre, on le lit dans sa config."""
    return {v.lower(): k for k, v in modele.config.id2label.items()}


@torch.inference_mode()
def probas_nli(tok, modele, premisses: list, hypotheses: list, batch: int = 64) -> np.ndarray:
    """Probabilités des 3 classes pour chaque paire (prémisse, hypothèse) ; forme (n, 3)."""
    res = []
    for i in range(0, len(premisses), batch):
        x = tok(premisses[i:i + batch], hypotheses[i:i + batch], truncation=True, max_length=256,
                padding=True, return_tensors="pt").to(modele.device)
        res.append(modele(**x).logits.softmax(-1).cpu().numpy())
    return np.concatenate(res)


def score_implication(tok, modele, textes: list, hypotheses: list, batch: int = 64) -> np.ndarray:
    """Pour chaque texte et chaque hypothèse : P(implication), l'hypothèse étant jugée seule (multi-label).
    Modèles à 3 classes : on ignore « neutre » et on garde P(implication) parmi {implication, contradiction}.
    Forme (n_textes, n_hypotheses)."""
    n, h = len(textes), len(hypotheses)
    premisses = [t for t in textes for _ in range(h)]
    p = probas_nli(tok, modele, premisses, hypotheses * n, batch)
    c = classes(modele)
    ent = p[:, c["entailment"]]
    if "contradiction" in c:
        ent = ent / (ent + p[:, c["contradiction"]])
    return ent.reshape(n, h)
