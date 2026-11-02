"""Partie G : latence d'inférence par exemple sur CPU et taille de chaque modèle.
À lancer sur une machine au repos. Résultats dans results/G_cout.csv."""

import json
import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (chargé par le pickle de TfidfLR)
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from valeval.baseline import TfidfLR, texte_complet
from valeval.data import label_matrix, load_split
from valeval.embeddings import charger_glove, etendre_avec_glove
from valeval.finaux import CACHE, MAX_LEN, RACINE
from valeval.hypotheses import liste
from valeval.models import BiLSTM, BiLSTMAttention, Etats, couche_embedding
from valeval.plm import MODELES, PLM, entrees
from valeval.preprocess import Vocabulaire, corpus
from valeval.zeroshot import MODELES as NLI

N = 100


def chrono(fonction, exemples) -> float:
    """Latence moyenne en millisecondes, un exemple à la fois ; le premier appel n'est pas compté."""
    fonction(exemples[0])
    debut = time.perf_counter()
    for e in exemples:
        fonction(e)
    return 1000 * (time.perf_counter() - debut) / len(exemples)


def taille_mo(objet) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        chemin = f.name
    if isinstance(objet, dict):
        torch.save(objet, chemin)
    else:
        with open(chemin, "wb") as g:
            pickle.dump(objet, g)
    taille = os.path.getsize(chemin) / 1e6
    os.remove(chemin)
    return taille


@torch.inference_mode()
def mesurer():
    tr, te = load_split("training"), load_split("test").head(N)
    y_tr = label_matrix(tr)
    lignes_te = [te.iloc[[i]] for i in range(len(te))]
    res = []

    # Baseline
    base = TfidfLR(C=1).fit(texte_complet(tr), y_tr)
    res.append({"modèle": "Baseline TF-IDF", "latence CPU (ms)": chrono(lambda d: base.predict_proba(texte_complet(d)), lignes_te),
                "taille (Mo)": taille_mo(base)})

    # C1 : latence mesurée avec l'architecture retenue ; les poids n'en changent pas le temps de calcul
    t_tr = corpus(tr)
    vocab = Vocabulaire(t_tr, min_freq=2)
    vecteurs = etendre_avec_glove(vocab, *charger_glove())
    c1 = BiLSTM(couche_embedding(len(vocab), 300, vecteurs, gele=True), pooling="max").eval()
    def lit_c1(d):
        ids = torch.as_tensor([vocab.encoder(corpus(d)[0], MAX_LEN)])
        return c1(ids, ids != 0).sigmoid()
    res.append({"modèle": "C1 BiLSTM + GloVe", "latence CPU (ms)": chrono(lit_c1, lignes_te), "taille (Mo)": taille_mo(c1.state_dict())})

    # D1, D2 : roberta-base + petite tête
    tok = AutoTokenizer.from_pretrained(MODELES["roberta"])
    enc = AutoModel.from_pretrained(MODELES["roberta"], dtype=torch.float32).eval()
    tete_d2 = BiLSTMAttention(Etats(768)).eval()
    lineaire = torch.nn.Linear(768, 20)

    def lit_enc(d):
        x = tok((d.conclusion + tok.sep_token + d.stance + tok.sep_token + d.premise).tolist(),
                truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return enc(**x).last_hidden_state, x.attention_mask.bool()

    def lit_d1(d):
        e, m = lit_enc(d)
        return lineaire(e.mean(1)).sigmoid()

    def lit_d2(d):
        e, m = lit_enc(d)
        return tete_d2(e, m).sigmoid()

    t_enc = taille_mo(enc.state_dict())
    res.append({"modèle": "D1 RoBERTa gelé + LR", "latence CPU (ms)": chrono(lit_d1, lignes_te), "taille (Mo)": t_enc + taille_mo(lineaire.state_dict())})
    res.append({"modèle": "D2 RoBERTa gelé + tête", "latence CPU (ms)": chrono(lit_d2, lignes_te), "taille (Mo)": t_enc + taille_mo(tete_d2.state_dict())})

    # E1, E2 : poids réels
    for titre, fichier, multi in [("E1 RoBERTa fine-tuné", "E1_roberta_ep8", False), ("E2 multi-tâche (λ=0,3)", "E2_roberta_lam0.3_ep8", True)]:
        m = PLM(MODELES["roberta"], multitache=multi).eval()
        m.load_state_dict(torch.load(CACHE / "ckpt" / f"{fichier}_seed42.pt"))

        def lit_e(d, m=m):
            ids = torch.as_tensor(entrees(d, tok))
            return m(ids, ids != tok.pad_token_id)[0].sigmoid()

        res.append({"modèle": titre, "latence CPU (ms)": chrono(lit_e, lignes_te), "taille (Mo)": taille_mo(m.state_dict())})

    # NLI : 20 paires (argument, hypothèse) par exemple ; 20 exemples suffisent vu le coût
    tok_n = AutoTokenizer.from_pretrained(NLI["deberta_large"])
    nli = AutoModelForSequenceClassification.from_pretrained(NLI["deberta_large"], dtype=torch.float32).eval()
    hyp = liste("en")

    def lit_nli(d):
        texte = texte_complet(d).tolist() * 20
        x = tok_n(texte, hyp, truncation=True, max_length=256, padding=True, return_tensors="pt")
        return nli(**x).logits.softmax(-1)

    res.append({"modèle": "B1 NLI", "latence CPU (ms)": chrono(lit_nli, lignes_te[:20]), "taille (Mo)": taille_mo(nli.state_dict())})

    pd.DataFrame(res).to_csv(RACINE / "results" / "G_cout.csv", index=False)
    print(pd.DataFrame(res).round(1).to_string(index=False))
    print("threads torch :", torch.get_num_threads())


if __name__ == "__main__":
    mesurer()
