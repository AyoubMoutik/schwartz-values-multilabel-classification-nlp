"""Partie F : les modèles finaux, avec une même interface probas(df)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from valeval.baseline import TfidfLR, texte_complet
from valeval.data import label_matrix, load_split
from valeval.embeddings import charger_glove, etendre_avec_glove
from valeval.hypotheses import liste
from valeval.models import BiLSTM, BiLSTMAttention, Etats, couche_embedding
from valeval.plm import MODELES, PLM, entrees, probas as probas_plm
from valeval.preprocess import Vocabulaire, corpus
from valeval.train import entrainer, fixer_graines, lots_etats, predire_probas
from valeval.zeroshot import charger, score_implication

RACINE = Path(__file__).resolve().parents[2]
CACHE = RACINE / "cache"
MAX_LEN = 128


class Baseline:
    nom, graines = "baseline_tfidf_lr", (0,)

    def __init__(self, tr, y_tr, va, y_va, graine):
        self.modele = TfidfLR(C=1).fit(texte_complet(tr), y_tr)

    def probas(self, df):
        return self.modele.predict_proba(texte_complet(df))


class C1:
    nom, graines = "C1_bilstm_glove", (42, 43, 44)
    glove = None

    def __init__(self, tr, y_tr, va, y_va, graine):
        C1.glove = C1.glove or charger_glove()
        t_tr = corpus(tr)
        self.vocab = Vocabulaire(t_tr, min_freq=2)
        vecteurs = etendre_avec_glove(self.vocab, *C1.glove)
        ids_tr, ids_va = self.encoder(tr), self.encoder(va)
        fixer_graines(graine)
        self.modele = BiLSTM(couche_embedding(len(self.vocab), 300, vecteurs, gele=True), pooling="max")
        entrainer(self.modele, ids_tr, y_tr, ids_va, y_va)

    def encoder(self, df):
        return [self.vocab.encoder(t, MAX_LEN) for t in corpus(df)]

    def probas(self, df):
        return predire_probas(self.modele, self.encoder(df), np.zeros((len(df), 20)))


class Extracteur:
    """roberta-base gelé : états de tous les tokens."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODELES["roberta"])
        self.modele = AutoModel.from_pretrained(MODELES["roberta"], dtype=torch.float32).cuda().eval()

    @torch.inference_mode()
    def etats(self, textes: list):
        res, masques = [], []
        for i in range(0, len(textes), 64):
            x = self.tok(textes[i:i + 64], truncation=True, max_length=MAX_LEN, padding="max_length",
                         return_tensors="pt").to("cuda")
            with torch.autocast("cuda"):
                res.append(self.modele(**x).last_hidden_state.half().cpu())
            masques.append(x.attention_mask.bool().cpu())
        return torch.cat(res), torch.cat(masques)

    def arguments(self, df):
        sep = self.tok.sep_token
        return self.etats((df.conclusion + sep + df.stance + sep + df.premise).tolist())

    @staticmethod
    def moyenne(etats, masques) -> np.ndarray:
        res = []
        for i in range(0, len(etats), 512):
            e, m = etats[i:i + 512].float(), masques[i:i + 512].unsqueeze(-1)
            res.append(((e * m).sum(1) / m.sum(1)).numpy())
        return np.concatenate(res)


class D1:
    nom, graines = "D1_roberta_gele_lr", (0,)

    def __init__(self, tr, y_tr, va, y_va, graine):
        self.ext = Extracteur()
        x = Extracteur.moyenne(*self.ext.arguments(tr))
        self.echelle = StandardScaler().fit(x)
        self.clf = OneVsRestClassifier(LogisticRegression(C=0.0003, class_weight="balanced", max_iter=2000))
        self.clf.fit(self.echelle.transform(x), y_tr)

    def probas(self, df):
        return self.clf.predict_proba(self.echelle.transform(Extracteur.moyenne(*self.ext.arguments(df))))


class D2:
    nom, graines = "D2_roberta_gele_bilstm_attention", (42, 43, 44)

    def __init__(self, tr, y_tr, va, y_va, graine):
        self.ext = Extracteur()
        d_tr, d_va = self.ext.arguments(tr), self.ext.arguments(va)
        fixer_graines(graine)
        self.modele = BiLSTMAttention(Etats(768))
        entrainer(self.modele, d_tr, y_tr, d_va, y_va, fabrique=lots_etats)

    def probas(self, df):
        return predire_probas(self.modele, self.ext.arguments(df), np.zeros((len(df), 20)), fabrique=lots_etats)


class Fine:
    """Poids sauvegardés en Partie E."""
    fichier, multitache = "", False

    def __init__(self, tr, y_tr, va, y_va, graine):
        self.tok = AutoTokenizer.from_pretrained(MODELES["roberta"])
        self.modele = PLM(MODELES["roberta"], multitache=self.multitache)
        self.modele.load_state_dict(torch.load(CACHE / "ckpt" / f"{self.fichier}_seed{graine}.pt"))
        self.modele.cuda()

    def probas(self, df):
        return probas_plm(self.modele, entrees(df, self.tok), self.tok.pad_token_id)


class E1(Fine):
    nom, graines, fichier = "E1_roberta_ep8", (42, 43, 44), "E1_roberta_ep8"


class E2(Fine):
    nom, graines, fichier, multitache = "E2_roberta_lam0.3_ep8", (42, 43, 44), "E2_roberta_lam0.3_ep8", True


class NLI:
    """Pas d'entraînement ; seuil par valeur, seule règle qui lui apporte un gain réel (Partie B1)."""
    nom, graines = "B1_nli_deberta_large", (0,)
    seuil_par_valeur = True

    def __init__(self, tr, y_tr, va, y_va, graine):
        self.tok, self.modele = charger("deberta_large")

    def probas(self, df):
        return score_implication(self.tok, self.modele, texte_complet(df).tolist(), liste("en"), 64)


def ecrire_predictions(chemin: Path, df: pd.DataFrame, p: np.ndarray, pred: np.ndarray, colonnes: list) -> None:
    """id, y_true, y_pred (20 chiffres 0/1, dans l'ordre de VALUES), puis une probabilité par valeur."""
    y = label_matrix(df)
    sortie = pd.DataFrame({"id": df.id.values, "y_true": ["".join(map(str, l)) for l in y],
                           "y_pred": ["".join(map(str, l)) for l in pred.astype(int)]})
    sortie = pd.concat([sortie, pd.DataFrame(p, columns=colonnes).round(5)], axis=1)
    sortie.to_csv(chemin, index=False)


def splits_finaux() -> dict:
    noms = ["test", "test-nahjalbalagha"]
    if (RACINE / "data" / "raw" / "arguments-test-nyt.tsv").exists():
        noms.append("test-nyt")
    return {n: load_split(n) for n in noms}
