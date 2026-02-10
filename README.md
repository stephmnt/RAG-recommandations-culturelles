# POC RAG Puls-Events (Etapes 1 et 2)

Ce depot couvre uniquement:
- Etape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
- Etape 2: recuperation + nettoyage OpenAgenda pour produire un dataset pret a indexer

Le scope volontairement exclu pour l'instant:
- index FAISS final
- chaines RAG LangChain
- API FastAPI/Flask
- evaluation RAGAS

## Choix techniques (lead-level)

- Gestion d'environnement: `requirements.txt` avec versions fixes pour une reproduction simple et rapide chez un evaluateur (pas de dependances implicites).
- Portabilite: `faiss-cpu` (pas de `faiss-gpu`).
- Secrets: `.env` local uniquement, jamais versionne.
- Zone geographique cible: `Lyon Metropole` (ville + coordonnees + rayon configurable).
- Fenetre temporelle: historique de 365 jours + horizon a venir configurable (par defaut +90 jours).
- Sortie etape 2: dataset propre `events_processed.parquet` avec `document_text` et `retrieval_metadata` prets pour l'etape 3.

## Prerequis

- Python 3.10 ou 3.11 recommande (minimum 3.8)
- `pip` recent

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Important:
- Python 3.13 n'est pas supporte par ce lock de dependances (notamment `pyarrow==17.0.0`).
- Utiliser Python 3.10 ou 3.11 pour une installation reproductible.

## Configuration des secrets

1. Copier l'exemple :
```bash
cp .env.example .env
```
2. Renseigner `OPENAGENDA_API_KEY` dans `.env`.
3. Ne jamais commiter `.env` (deja ignore via `.gitignore`).

## Smoke test

Verifier que les imports critiques et versions sont corrects:

```bash
python3 scripts/check_env.py
```

Le script:
- affiche les versions de Python, langchain, faiss, mistral/mistralai, pandas, requests
- teste ces imports:
  - `import faiss`
  - `from langchain.vectorstores import FAISS`
  - `from langchain.embeddings import HuggingFaceEmbeddings`
  - `from mistral import MistralClient`
- retourne un code non-zero si un import echoue

## Recuperer les donnees OpenAgenda

La configuration est centralisee dans `config.yaml`:
- zone (Lyon, coordonnees, rayon)
- periode (`start_date` et `end_date`)
- pagination (`page_size`, `max_pages`, `max_events`)
- langue (`fr`)

Si `start_date`/`end_date` sont vides, `scripts/build_dataset.py` applique:
- `start_date = date_du_jour - 365 jours`
- `end_date = date_du_jour + 90 jours`

## Construire le dataset

```bash
python3 scripts/build_dataset.py --config config.yaml
```

Sorties produites:
- `data/raw/events_raw.jsonl`
- `data/processed/events_processed.parquet`
- `logs/build_dataset.log`

Le script affiche un resume:
- nombre d'evenements recuperes
- nombre apres filtres temporels
- nombre de doublons supprimes
- nombre d'enregistrements invalides
- nombre final exporte

## Ou sont les fichiers generes ?

- Bruts: `data/raw/`
- Nettoyes: `data/processed/`
- Logs: `logs/`
- Exemple versionne: `data/sample/events_sample.jsonl`

`data/` local est ignore par git, sauf `data/sample/`.

## Lancer les tests

```bash
pytest -q
```

Les tests sont relancables et 100% offline (HTTP mocke).

## Notebook de validation (Etapes 1-2)

Notebook simple et pedagogique:
- `notebooks/validation_etapes_1_2.ipynb`

Il verifie:
- smoke test environnement (versions + imports critiques)
- configuration OpenAgenda
- pagination client en mode mock (sans reseau)
- cleaning + validation schema
- simulation ecriture raw/processed
- execution optionnelle de `scripts/check_env.py` et `pytest -q`

## Depannage

- Cle API absente:
  - symptome: erreur 401/403
  - action: verifier `.env` et `OPENAGENDA_API_KEY`
- 401/403:
  - verifier validite de la cle et quotas OpenAgenda
- Timeouts:
  - augmenter `request.timeout_seconds` dans `config.yaml`
  - reduire `pagination.page_size`
- Pagination incomplete:
  - augmenter `pagination.max_pages` ou `pagination.max_events`
  - verifier le parametre `city`/filtres geographiques

## Structure du depot (Etapes 1-2)

```text
.
├── .env.example
├── .gitignore
├── README.md
├── config.yaml
├── requirements.txt
├── data/
│   └── sample/
│       └── events_sample.jsonl
├── logs/
│   └── .gitkeep
├── mistral/
│   └── __init__.py
├── notebooks/
│   └── validation_etapes_1_2.ipynb
├── scripts/
│   ├── build_dataset.py
│   └── check_env.py
├── src/
│   ├── __init__.py
│   ├── openagenda/
│   │   ├── __init__.py
│   │   └── client.py
│   └── preprocess/
│       ├── __init__.py
│       ├── cleaning.py
│       └── schema.py
└── tests/
    ├── conftest.py
    ├── test_cleaning.py
    ├── test_client.py
    └── test_schema.py
```

## Reproductibilite

- Dependances figees dans `requirements.txt`.
- Aucun secret versionne.
- Aucun fichier genere versionne (hors echantillon `data/sample/`).
- Pipeline parametrable via `config.yaml` + variables d'environnement.
