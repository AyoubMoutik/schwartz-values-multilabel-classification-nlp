"""Reconstruit arguments-test-nyt.tsv depuis l'Internet Archive.

Version allégée du nyt-downloader des organisateurs (même découpage, sans navigateur).
Usage : uv run --with requests --with beautifulsoup4 python scripts/download_nyt.py
Le fichier nyt01-spans.tsv vient de touche-webis-de/touche-code (semeval23/human-value-detection/nyt-downloader)."""

import hashlib
import html
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

RACINE = Path(__file__).resolve().parents[1]
OUTIL = RACINE / "cache" / "nyt_outil"
PAGES = OUTIL / "pages"
SORTIE = RACINE / "data" / "raw" / "arguments-test-nyt.tsv"
SHA256 = "fb02505106eaeb7ad5f2f1eaa54346a4286270d95707e8414eeb02649964615b"


def telecharger(url: str) -> str:
    PAGES.mkdir(parents=True, exist_ok=True)
    fichier = PAGES / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if fichier.exists():
        return fichier.read_text(encoding="utf-8")
    for essai in range(6):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            fichier.write_text(r.text, encoding="utf-8")
            time.sleep(45)
            return r.text
        except requests.RequestException as e:
            print("nouvel essai :", url, e, flush=True)
            time.sleep(120 * (essai + 1))
    raise RuntimeError(f"échec : {url}")


def contenu(page: str) -> str:
    """Titre, chapeau puis paragraphes, comme parse_article_content des organisateurs."""
    soup = BeautifulSoup(unicodedata.normalize("NFKC", html.unescape(page)), "html.parser")
    titre = soup.find("h1", attrs={"data-testid": "headline"})
    chapeau = soup.find("p", attrs={"id": "article-summary"})
    fragments = [x.text.strip() for x in (titre, chapeau) if x is not None and x.text.strip()]
    corps = soup.find("section", attrs={"name": "articleBody"})
    fragments.append(" ".join(p.text.strip() for p in corps.find_all("p") if p.text.strip()))
    return " ".join(fragments)


def main():
    df = pd.read_csv(OUTIL / "nyt01-spans.tsv", sep="\t")
    urls = "https://web.archive.org/web/" + df["Internet Archive timestamp"].astype(str) + "/" + df["URL"]
    textes = {}
    for u in sorted(set(urls)):
        print("article", u, flush=True)
        textes[u] = contenu(telecharger(u))
    lignes = []
    for u, l in zip(urls, df.itertuples()):
        def extrait(position):
            debut, fin = map(int, position.split(":"))
            return textes[u][debut:fin]
        lignes.append({"Argument ID": l[1], "Conclusion": extrait(df.loc[l.Index, "Conclusion token position"]),
                       "Stance": df.loc[l.Index, "Stance"], "Premise": extrait(df.loc[l.Index, "Premise token position"])})
    pd.DataFrame(lignes).to_csv(SORTIE, sep="\t", index=False, lineterminator="\n")
    empreinte = hashlib.sha256(SORTIE.read_bytes()).hexdigest()
    print("sha256 conforme :", empreinte == SHA256, empreinte, flush=True)


if __name__ == "__main__":
    main()
