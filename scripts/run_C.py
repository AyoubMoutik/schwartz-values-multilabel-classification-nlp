"""Partie C : entraîne les réseaux demandés. Usage : python scripts/run_C.py C1_glove_max C2_glove@43 (@graine, 42 par défaut)"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from valeval.data import RAW, VALUES, label_matrix, load_split
from valeval.embeddings import charger_glove, etendre_avec_glove
from valeval.evaluate import log_result, scores, seuil_unique
from valeval.models import BiLSTM, BiLSTMAttention, CharCNN, MiniTransformer, couche_embedding
from valeval.keywords import depuis_descriptions
from valeval.preprocess import (NEGATIONS, PONCTUATION, Vocabulaire, VocabulaireCaracteres, corpus, retirer,
                                texte_caracteres)
from valeval.train import entrainer, fixer_graines, lots, poids_positifs, tirage_surechantillonne

SORTIES = Path(__file__).resolve().parents[1] / "cache" / "runs"
MAX_LEN, MAX_CHARS, DIM = 128, 512, 300

# nom : (architecture, embeddings, options)
EXPERIENCES = {
    "C1_aleatoire_dernier": ("bilstm", "aleatoire", {"pooling": "dernier"}),
    "C1_aleatoire_max": ("bilstm", "aleatoire", {"pooling": "max"}),
    "C1_glove_dernier": ("bilstm", "glove", {"pooling": "dernier"}),
    "C1_glove_max": ("bilstm", "glove", {"pooling": "max"}),
    "C2_glove": ("attention", "glove", {}),
    "C3_glove_moyenne": ("transformer", "glove", {"pooling": "moyenne"}),
    "C3_glove_max": ("transformer", "glove", {"pooling": "max"}),
    "C3_glove_max_sans_pe": ("transformer", "glove", {"pooling": "max", "positionnel": False}),
    "C3_glove_moyenne_sans_pe": ("transformer", "glove", {"pooling": "moyenne", "positionnel": False}),
    "C4_charcnn": ("char", "caracteres", {}),
    "C1_perte_ponderee": ("bilstm", "glove", {"pooling": "max", "desequilibre": "perte"}),
    "C1_surechantillonnage": ("bilstm", "glove", {"pooling": "max", "desequilibre": "tirage"}),
    "C1_sans_negations": ("bilstm", "glove", {"pooling": "max", "filtre": "negations"}),
    "C1_sans_mots_vides": ("bilstm", "glove", {"pooling": "max", "filtre": "mots_vides"}),
    "C1_sans_ponctuation": ("bilstm", "glove", {"pooling": "max", "filtre": "ponctuation"}),
    "C1_sans_mots_moraux": ("bilstm", "glove", {"pooling": "max", "filtre": "mots_moraux"}),
}


def mots_a_retirer(filtre):
    if filtre == "negations":
        return NEGATIONS
    if filtre == "mots_vides":
        return set(ENGLISH_STOP_WORDS) | NEGATIONS
    if filtre == "ponctuation":
        return PONCTUATION
    dico = depuis_descriptions(RAW / "value-categories.json")
    return {m for v in VALUES for m in dico[v]}


def construire(archi, embedding, options):
    return {"bilstm": BiLSTM, "attention": BiLSTMAttention, "transformer": MiniTransformer}[archi](embedding, **options)


@torch.inference_mode()
def sauver_attention(modele, tokens, ids, y, chemin):
    """Pour chaque argument de val : ses tokens et le poids d'attention de chacun."""
    modele.eval()
    alphas = []
    for x, m, _ in lots(ids, y, 256, False):
        a = modele.poids(x, m).cpu().numpy()
        alphas += [ligne[:int(n)].tolist() for ligne, n in zip(a, m.sum(1).cpu())]
    chemin.write_text(json.dumps([{"tokens": t[:len(p)], "alpha": p} for t, p in zip(tokens, alphas)]), encoding="utf-8")


def executer(nom, graine=42):
    archi, emb, options = EXPERIENCES[nom]
    options = dict(options)
    desequilibre = options.pop("desequilibre", None)
    filtre = options.pop("filtre", None)
    tr, va = load_split("training"), load_split("validation")
    y_tr, y_va = label_matrix(tr), label_matrix(va)
    if archi == "char":
        t_tr = [texte_caracteres(l) for l in tr.itertuples()]
        t_va = [texte_caracteres(l) for l in va.itertuples()]
        vocab = VocabulaireCaracteres(t_tr)
        ids_tr = [vocab.encoder(t, MAX_CHARS) for t in t_tr]
        ids_va = [vocab.encoder(t, MAX_CHARS) for t in t_va]
        fixer_graines(graine)
        modele = CharCNN(len(vocab), **options)
    else:
        t_tr, t_va = corpus(tr), corpus(va)
        if filtre:
            t_tr, t_va = retirer(t_tr, mots_a_retirer(filtre)), retirer(t_va, mots_a_retirer(filtre))
        vocab = Vocabulaire(t_tr, min_freq=2)
        vecteurs = None
        if emb == "glove":
            vecteurs = etendre_avec_glove(vocab, *GLOVE())
        ids_tr = [vocab.encoder(t, MAX_LEN) for t in t_tr]
        ids_va = [vocab.encoder(t, MAX_LEN) for t in t_va]
        fixer_graines(graine)
        modele = construire(archi, couche_embedding(len(vocab), DIM, vecteurs, gele=True), options)
    res = entrainer(modele, ids_tr, y_tr, ids_va, y_va,
                    pos_weight=poids_positifs(y_tr) if desequilibre == "perte" else None,
                    tirage=tirage_surechantillonne(y_tr, graine) if desequilibre == "tirage" else None)

    p = res.pop("probas_val")
    s = seuil_unique(y_va, p)
    val = scores(y_va, p >= s)
    log_result(nom, f"mots minuscules, embeddings {emb}", graine, val=val, temps_s=round(res["temps_s"], 1),
               params=res["params"], notes=f"seuil unique {s:.2f}, {len(res['courbes'])} époques")
    SORTIES.mkdir(parents=True, exist_ok=True)
    if archi == "attention":
        sauver_attention(modele, t_va, ids_va, y_va, SORTIES / f"{nom}_seed{graine}_attention.json")
    np.save(SORTIES / f"{nom}_seed{graine}_val.npy", p)
    (SORTIES / f"{nom}_seed{graine}.json").write_text(json.dumps({**res, "seuil": s, "macro_f1": val["macro_f1"]}), encoding="utf-8")
    print(f"{nom}: macro-F1 val {val['macro_f1']:.3f} (seuil {s:.2f}), {len(res['courbes'])} époques, "
          f"{res['temps_s']:.0f}s, {res['params']:,} paramètres entraînés", flush=True)


_glove = None


def GLOVE():
    global _glove
    if _glove is None:
        _glove = charger_glove()
    return _glove


if __name__ == "__main__":
    for demande in sys.argv[1:]:
        nom, _, graine = demande.partition("@")
        executer(nom, int(graine or 42))
        torch.cuda.empty_cache()
