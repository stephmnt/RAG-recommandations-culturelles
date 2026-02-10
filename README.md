# POC RAG Puls-Events (Étapes 1 à 3)

Ce dépôt couvre :

* Étape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
* Étape 2: récupération + nettoyage OpenAgenda pour produire un dataset prêt à indexer
* Étape 3: chunking + embeddings + indexation FAISS persistée

Le scope volontairement exclu pour l'instant :

* chaînes RAG LangChain
* API FastAPI/Flask
* évaluation RAGAS

## Choix techniques (lead-level)

* Gestion d'environnement: `requirements.txt` avec versions fixes pour une reproduction simple et rapide chez un évaluateur (pas de dépendances implicites).
* Portabilité: `faiss-cpu` (pas de `faiss-gpu`).
* Secrets: `.env` local uniquement, jamais versionné.
* Zone géographique cible: `Département de l'Hérault (34)` (filtre département + coordonnées/rayon autour de Montpellier).
* Fenêtre temporelle: historique de 365 jours + horizon à venir configurable (par défaut +90 jours).
* Sortie étape 2: dataset propre `events_processed.parquet` avec `document_text` et `retrieval_metadata` prêts pour l'étape 3.
* Étape 3:
  - chunking via `RecursiveCharacterTextSplitter`
  - embeddings par défaut HuggingFace, fallback depuis Mistral vers HuggingFace si indisponible
  - index FAISS persisté localement + métadonnées de build

## Prérequis

* Python 3.10 ou 3.11 recommandé (minimum 3.8)
* `pip` récent

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Important :

* Python 3.13 n'est pas supporté par ce lock de dépendances (notamment `pyarrow==17.0.0`).
* Utiliser Python 3.10 ou 3.11 pour une installation reproductible.

## Configuration des secrets

1. Copier l'exemple `cp .env.example .env`
2. Renseigner `OPENAGENDA_API_KEY` dans `.env`.
3. Ne jamais commiter `.env` (déjà ignoré via `.gitignore`).

## Smoke test

Vérifier que les imports critiques et versions sont corrects:

```bash
python3 scripts/check_env.py
```

Le script :

* affiche les versions de Python, langchain, faiss, mistral/mistralai, pandas, requests
* teste ces imports:

  * `import faiss`
  * `from langchain.vectorstores import FAISS`
  * `from langchain.embeddings import HuggingFaceEmbeddings`
  * `from mistral import MistralClient`
* retourne un code non-zero si un import échoue

## Récupérer les données OpenAgenda

La configuration est centralisée dans `config.yaml` :

* zone (département 34, ville pivot Montpellier, coordonnées, rayon)
* période (`start_date` et `end_date`)
* pagination (`page_size`, `max_pages`, `max_events`)
* langue (`fr`)

Si `start_date`/`end_date` sont vides, `scripts/build_dataset.py` applique :

* `start_date = date_du_jour - 365 jours`
* `end_date = date_du_jour + 90 jours`

## Construire le dataset

```bash
python3 scripts/build_dataset.py --config config.yaml
```

Sorties produites :

* `data/raw/events_raw.jsonl`
* `data/processed/events_processed.parquet`
* `logs/build_dataset.log`

Le script affiche un résumé :

* nombre d'événements récupérés
* nombre après filtres temporels
* nombre de doublons supprimés
* nombre d'enregistrements invalides
* nombre final exporté

## Où sont les fichiers générés ?

* Bruts: `data/raw/`
* Nettoyés: `data/processed/`
* Logs: `logs/`
* Exemple versionné: `data/sample/events_sample.jsonl`

`data/` local est ignoré par git, sauf `data/sample/`.

## Lancer les tests

```bash
pytest -q
```

Les tests sont relançables et 100% offline (HTTP mocké).

Pour inclure le smoke de performance Étape 3:

```bash
pytest -q -m slow
```

## Construire l'index FAISS

Build/rebuild complet depuis le dataset processed:

```bash
python3 scripts/build_index.py \
  --input data/processed/events_processed.parquet \
  --output artifacts/faiss_index
```

Par défaut le script lit aussi `configs/indexing.yaml`:
- chunking (`chunk_size`, `chunk_overlap`, `min_chunk_size`, `separators`)
- embeddings (`provider`, `huggingface_model`, `mistral_model`)
- FAISS (`normalize_L2`)

## Tester une recherche locale

```bash
python3 scripts/query_index.py --query "concert jazz montpellier ce week-end" --k 5
```

Affichage:
- score
- `event_id`
- `start_datetime`
- `city`
- `url`
- extrait du chunk

## Où sont stockés les artifacts ?

- Sortie index: `artifacts/faiss_index/`
- Fichiers FAISS:
  - `index.faiss`
  - `index.pkl`
  - `index_metadata.json`

Les gros binaires sont ignorés par git, la structure et les README restent versionnés.

## Rebuild de l'index (déterminisme et paramètres)

Le rebuild est reproductible via:
- dataset d'entrée explicite
- config explicite (`configs/indexing.yaml`)
- hash de dataset (`dataset_hash`) stocké dans `index_metadata.json`
- résumé des paramètres chunking/embeddings/FAISS dans `index_metadata.json`

## Compatibilité LangChain (imports robustes)

Le code gère les variations selon version:
- `langchain_community.vectorstores.FAISS` puis fallback `langchain.vectorstores.FAISS`
- embeddings HF/Mistral avec fallback en cas d'indisponibilité

## Dépannage Étape 3

- `ImportError` sur FAISS / LangChain:
  - vérifier `pip install -r requirements.txt`
  - vérifier version Python 3.10/3.11
- `EMBEDDING_PROVIDER=mistral` sans clé:
  - fallback automatique vers HuggingFace avec warning log
- modèle HF indisponible:
  - définir `EMBEDDING_MODEL` ou `embeddings.huggingface_model` dans `configs/indexing.yaml`
- erreur de load FAISS:
  - reconstruire l'index (`scripts/build_index.py`) puis relancer la requête

## Notebook de validation (Étapes 1-2)

Notebook simple et pédagogique :

* `notebooks/validation_etapes_1_2.ipynb`

Il vérifie :

* smoke test environnement (versions + imports critiques)
* configuration OpenAgenda
* pagination client en mode mock (sans réseau)
* cleaning + validation schéma
* simulation écriture raw/processed
* exécution optionnelle de `scripts/check_env.py` et `pytest -q`

## Dépannage

* Clé API absente:

  * symptôme: erreur 401/403
  * action: vérifier `.env` et `OPENAGENDA_API_KEY`
* 401/403:

  * vérifier validité de la clé et quotas OpenAgenda
* Timeouts:

  * augmenter `request.timeout_seconds` dans `config.yaml`
  * réduire `pagination.page_size`
* Pagination incomplète:

  * augmenter `pagination.max_pages` ou `pagination.max_events`
  * vérifier les paramètres `department` et `city`/filtres géographiques

## Structure du dépôt (Étapes 1-3)

```text
.
├── .env.example
├── .gitignore
├── README.md
├── config.yaml
├── configs/
│   └── indexing.yaml
├── requirements.txt
├── artifacts/
│   ├── README.md
│   └── faiss_index/
│       └── README.md
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
│   ├── build_index.py
│   ├── check_env.py
│   └── query_index.py
├── src/
│   ├── __init__.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── build_index.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── search.py
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
    ├── test_indexing_chunking.py
    ├── test_indexing_pipeline.py
    └── test_schema.py
```

## Reproductibilité

* Dépendances figées dans `requirements.txt`.
* Aucun secret versionné.
* Aucun fichier généré versionné (hors échantillon `data/sample/`).
* Pipeline paramétrable via `config.yaml` + variables d'environnement.
