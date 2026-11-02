# TP2 — Identifier les valeurs humaines derrière un argument

Module Transformers & LLMs (Prof. Yassir Matrane, ENSAM Casablanca) · Groupe 11 · Jamila Ouzzane & Ayoub Moutik

Pour un argument (conclusion, position pour / contre, prémisse), prédire les **valeurs humaines** qu'il mobilise parmi les 20 catégories de Schwartz. C'est une classification **multi-label** : chaque argument peut avoir plusieurs valeurs. Jeu de données : **Touché23-ValueEval** (SemEval-2023 Tâche 4). Métrique principale : **macro-F1** sur les 20 valeurs.

## Résultat principal

Modèle retenu : **`roberta-base` fine-tuné** (8 époques, seuil unique 0,52 choisi sur val).

| Modèle | macro-F1 val | macro-F1 test | Nahj al-Balagha | NYT | Latence CPU |
|---|---|---|---|---|---|
| Plancher « toutes les valeurs » | 0,272 | 0,248 | 0,122 | 0,119 | – |
| Baseline TF-IDF + régression logistique | 0,394 | 0,368 | 0,178 | 0,145 | 3,5 ms |
| B1 NLI zero-shot (DeBERTa-v3-large) | 0,407¹ | 0,405 | 0,214 | 0,170 | 2 809 ms |
| C1 BiLSTM + GloVe | 0,415 | 0,394 | 0,166 | 0,176 | 3,1 ms |
| D1 RoBERTa gelé + régression logistique | 0,446 | 0,436 | 0,200 | 0,229 | 34 ms |
| **E1 RoBERTa fine-tuné** | **0,483** | **0,473** | **0,232** | **0,229** | 37 ms |

Moyennes sur 3 graines quand le modèle en a. Le test, Nahj al-Balagha et NYT n'ont été utilisés qu'en Partie F, une fois tous les choix figés sur val.
¹ Score en validation croisée sur val ; avec les seuils réglés sur tout val, 0,427.

Avec la définition du F1 de la tâche partagée (moyenne harmonique de la précision et du rappel macro), E1 obtient ≈ 0,48 sur le test, contre 0,42 pour la baseline BERT des organisateurs et 0,56 pour la meilleure équipe (Kiesel et al., 2023).

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et Python 3.11 (uv l'installe). Un GPU NVIDIA est nécessaire pour les Parties C à E (testé sur RTX 4070 Laptop, 8 Go) ; l'analyse (F, G) tourne sur CPU.

```bash
uv sync                                             # environnement figé par uv.lock ; torch CUDA 12.8
uv run python scripts/telecharger_donnees.py        # Zenodo (vérif. MD5), GloVe, positions NYT
uv run --with requests --with beautifulsoup4 python scripts/download_nyt.py   # textes NYT (Internet Archive)
uv run pytest -q
```

Les textes NYT ne sont pas distribués pour des raisons de droits ; `download_nyt.py` les reconstruit depuis l'Internet Archive et vérifie le SHA-256 fourni par les organisateurs. Si l'Archive limite le débit (erreur 429), relancer plus tard : les pages déjà téléchargées sont gardées en cache.

## Reproduire, partie par partie

| Partie | Commande | Notebook |
|---|---|---|
| 4.1 Audit | – | `notebooks/01_audit.ipynb` |
| A. Distributions, classifieurs triviaux, baseline, raccourcis | – | `notebooks/02_partieA.ipynb` |
| B1. NLI zero-shot | `uv run python scripts/run_zeroshot.py` | `notebooks/03_partieB.ipynb` |
| B2. Dictionnaire de mots-clés | – | `notebooks/04_partieB2.ipynb` |
| C. BiLSTM, attention, mini-Transformer, char-CNN | `uv run python scripts/run_C.py C1_glove_max@42 …` (noms dans `EXPERIENCES`) | `notebooks/05_partieC.ipynb` |
| C. Casse (baseline) | `uv run python scripts/run_tfidf_casse.py` | idem |
| D. RoBERTa gelé | `uv run python scripts/run_D.py extraire d1 d2@42 d2@43 d2@44` | `notebooks/06_partieD.ipynb` |
| E. Fine-tuning | `uv run python scripts/run_E.py roberta@42~8 deberta@42~8 roberta@42:0.3~8 …` (`modèle@graine:λ~époques`) | `notebooks/07_partieE.ipynb` |
| F. Test, hors domaine, F3 à F6 | `uv run python scripts/run_F.py` | `notebooks/08_partieF.ipynb` |
| G. Latence et taille sur CPU | `uv run python scripts/run_G.py` | `notebooks/09_partieG.ipynb` |

Les notebooks lisent les scores calculés par les scripts dans `cache/`, qui n'est pas versionné (~10 Go : poids, états RoBERTa, GloVe). Les notebooks 02, 03 et 04 ajoutent des lignes au journal à chaque exécution.

## Organisation

```
src/valeval/     data · preprocess · baseline · keywords · hypotheses · zeroshot
                 embeddings · models · train · plm · evaluate · finaux
scripts/         une commande par partie (voir ci-dessus)
notebooks/       01 à 09 : analyses, figures et lectures
results/         results.csv (journal de chaque exécution), tableaux finaux, F6_erreurs.csv
predictions/     prédictions du test de chaque modèle final : <modèle>_seed<k>.csv
figures/         figures du compte-rendu
tests/           tests unitaires (tokenisation, vocabulaire)
rapport/         compte-rendu
```

Format de `predictions/<modèle>_seed<k>.csv` : `id`, `y_true` et `y_pred` (20 chiffres 0/1, dans l'ordre des colonnes de `labels-test.tsv`), puis une probabilité par valeur.

## Règles suivies

- Split officiel train / val / test ; aucune conclusion n'est commune aux splits (vérifié).
- Toutes les décisions (prétraitement, architecture, `C`, époques, seuils) sont prises sur **val** ; le test est resté scellé jusqu'à la Partie F.
- Toute transformation apprise (vocabulaire, TF-IDF, mise à l'échelle, poids de classes, sur-échantillonnage) est ajustée sur le **train** uniquement.
- Graines fixées (`random`, `numpy`, `torch`) ; 3 graines (42, 43, 44) pour les modèles finaux.
- Chaque exécution est journalisée dans `results/results.csv`.

## Référence

Kiesel, J., Alshomary, M., Mirzakhmedova, N., Heinrich, M., Handke, N., Wachsmuth, H. et Stein, B. (2023). *SemEval-2023 Task 4: ValueEval: Identification of Human Values Behind Arguments.* SemEval-2023, p. 2287–2303. https://aclanthology.org/2023.semeval-1.313

Données : https://doi.org/10.5281/zenodo.6814563 (CC BY-SA 4.0).
