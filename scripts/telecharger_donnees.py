"""Télécharge tout ce que le projet ne versionne pas.

1. Touché23-ValueEval depuis Zenodo (version 2024-01-24) vers data/raw/, avec vérification MD5.
2. GloVe 6B 300d (dépôt Hugging Face stanfordnlp/glove) vers cache/glove/.
3. Le fichier des positions des arguments NYT (touche-code) vers cache/nyt_outil/.
   Les textes NYT se reconstruisent ensuite avec scripts/download_nyt.py.

Usage : uv run python scripts/telecharger_donnees.py"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

RACINE = Path(__file__).resolve().parents[1]
RAW = RACINE / "data" / "raw"
ZENODO = "https://zenodo.org/api/records/10564870"
IGNORES = {"annotations-level1.tsv"}          # 53 Mo de votes individuels, inutiles ici
NYT = ("https://raw.githubusercontent.com/touche-webis-de/touche-code/main/"
       "semeval23/human-value-detection/nyt-downloader/")


def md5(chemin: Path) -> str:
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def zenodo():
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO, timeout=60) as r:
        fichiers = json.load(r)["files"]
    for f in fichiers:
        if f["key"] in IGNORES:
            continue
        cible = RAW / f["key"]
        attendu = f["checksum"].removeprefix("md5:")
        if cible.exists() and md5(cible) == attendu:
            print("déjà présent :", f["key"])
            continue
        urllib.request.urlretrieve(f["links"]["self"], cible)
        assert md5(cible) == attendu, f"MD5 incorrect pour {f['key']}"
        print("téléchargé :", f["key"])


def glove():
    cible = RACINE / "cache" / "glove" / "glove.6B.300d.txt"
    if cible.exists():
        print("GloVe déjà présent")
        return
    cible.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(hf_hub_download("stanfordnlp/glove", "glove.6B.zip")) as z:
        z.extract("glove.6B.300d.txt", cible.parent)
    print("GloVe extrait :", cible)


def positions_nyt():
    dossier = RACINE / "cache" / "nyt_outil"
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in ["nyt01-spans.tsv", "arguments-test-nyt.sha256"]:
        if not (dossier / nom).exists():
            urllib.request.urlretrieve(NYT + nom, dossier / nom)
    print("positions NYT prêtes ; lancer : uv run --with requests --with beautifulsoup4 python scripts/download_nyt.py")


if __name__ == "__main__":
    zenodo()
    glove()
    positions_nyt()
