# POC RAG Puls-Events (Etapes 1 a 4)

Ce depot couvre:

- Etape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
- Etape 2: recuperation + nettoyage OpenAgenda pour produire un dataset pret a indexer
- Etape 3: chunking + embeddings + indexation FAISS persistante
- Etape 4: moteur RAG (retrieval + generation Mistral) sans API web

Hors scope actuel:

- API FastAPI/Flask (etape 5)
- dockerisation
- evaluation RAGAS complete

## Choix techniques (lead-level)

- Gestion d'environnement: `requirements.txt` avec versions fixes.
- Portabilite: `faiss-cpu` (pas de `faiss-gpu`).
- Secrets: `.env` local uniquement, jamais versionne.
- Zone geographique cible: `Departement de l'Herault (34)`.
- Fenetre temporelle: historique glissant 365 jours + horizon a venir configurable.
- Sortie etape 2: `events_processed.parquet` contenant `document_text` et `retrieval_metadata`.
- Etape 3:
  - chunking via `RecursiveCharacterTextSplitter`
  - embeddings HF par defaut, fallback Mistral -> HF
  - index FAISS persiste + metadata de build
- Etape 4:
  - retrieval top-k dedup par `event_id`
  - contexte structure et borne en taille
  - prompting FR anti-hallucination
  - sortie JSON-serialisable avec sources et latences

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

- Python 3.13 n'est pas supporte par ce lock (notamment `pyarrow==17.0.0`).
- Utiliser Python 3.10/3.11 pour une installation stable.

## Configuration des secrets

1. Copier l'exemple:

```bash
cp .env.example .env
```

2. Renseigner les cles dans `.env`:

- `OPENAGENDA_API_KEY` (etapes 2)
- `MISTRAL_API_KEY` (etape 4)

3. Ne jamais commiter `.env` (deja ignore dans `.gitignore`).

## Smoke test environnement (Etape 1)

```bash
python3 scripts/check_env.py
```

Le script:

- affiche les versions de Python, langchain, faiss, mistralai, pandas, requests
- verifie les imports critiques
- retourne un code non-zero si un import echoue

## Recuperer les donnees OpenAgenda (Etape 2)

La configuration est centralisee dans `config.yaml`:

- zone (departement 34 + ville pivot)
- periode (`start_date`, `end_date`)
- pagination (`page_size`, `max_pages`, `max_events`)
- langue (`fr`)

Si les dates sont vides, `scripts/build_dataset.py` applique:

- `start_date = aujourd'hui - 365 jours`
- `end_date = aujourd'hui + 90 jours`

### Construire le dataset

```bash
python3 scripts/build_dataset.py --config config.yaml
```

Sorties:

- `data/raw/events_raw.jsonl`
- `data/processed/events_processed.parquet`
- `logs/build_dataset.log`

## Construire l'index FAISS (Etape 3)

```bash
python3 scripts/build_index.py \
  --input data/processed/events_processed.parquet \
  --output artifacts/faiss_index
```

Configuration par defaut dans `configs/indexing.yaml`:

- chunking (`chunk_size`, `chunk_overlap`, `min_chunk_size`, `separators`)
- embeddings (`provider`, `huggingface_model`, `mistral_model`)
- FAISS (`normalize_L2`)

### Tester une recherche locale sur l'index

```bash
python3 scripts/query_index.py --query "concert jazz montpellier ce week-end" --k 5
```

## Moteur RAG (Etape 4)

Le moteur RAG est dans `src/rag/` et suit un flux single-turn:

1. chargement de l'index FAISS local,
2. retrieval top-k + deduplication par evenement,
3. construction d'un contexte borne,
4. generation via Mistral,
5. retour structure: reponse + sources + meta (latences, model, prompt).

### Configurer Mistral

```bash
export MISTRAL_API_KEY="votre_cle"
# Optionnel
export MISTRAL_MODEL="mistral-small-latest"
```

(ou renseigner ces variables dans `.env`).

### Executer une question en local

```bash
python scripts/ask_local.py --query "Quels evenements jazz a Montpellier cette semaine ?" --debug
```

Options utiles:

- `--top_k 6`
- `--index_path artifacts/faiss_index`
- `--prompt_version v1`

### Comprendre les sources

Chaque source retournee contient:

- `event_id`, `title`, `start_datetime`, `end_datetime`
- `city`, `location_name`, `url`
- `score` (si disponible)
- `snippet` (extrait court du chunk)

Le service limite les sources dedupliquees a `max_sources` (defaut 5).

### Evaluation smoke (sans RAGAS)

Jeu d'evaluation synthétique versionne:

- `data/eval/smoke_eval.jsonl`

Lancer l'evaluation:

```bash
python scripts/evaluate_smoke.py --offline
```

Sortie:

- rapport JSON: `reports/smoke_eval_report.json`
- resume console: taux URL attendues, overlap de mots-cles

## Ou sont stockes les artifacts ?

- Index FAISS: `artifacts/faiss_index/`
- Metadata de build index: `artifacts/faiss_index/index_metadata.json`
- Logs: `logs/`
- Rapport smoke: `reports/smoke_eval_report.json`

Les gros binaires et sorties runtime sont ignores par git.

## Rebuild de l'index (determinisme et parametres)

Le rebuild est reproductible via:

- dataset d'entree explicite
- config explicite (`configs/indexing.yaml`)
- hash dataset (`dataset_hash`) dans `index_metadata.json`
- resume des parametres chunking/embeddings/FAISS

## Compatibilite LangChain

Imports robustes selon versions:

- vectorstore: `langchain_community.vectorstores.FAISS` puis fallback `langchain.vectorstores.FAISS`
- embeddings HF/Mistral: fallback gere dans la factory

## Lancer les tests

Tests unitaires rapides (offline):

```bash
pytest -q
```

Inclure les tests lents performance:

```bash
pytest -q -m slow
```

## Depannage

### Etapes 1-2

- Cle OpenAgenda absente ou invalide (`401/403`): verifier `.env` + quotas.
- Timeouts OpenAgenda: augmenter `request.timeout_seconds`, reduire `page_size`.
- Pagination insuffisante: augmenter `max_pages`/`max_events`.

### Etape 3

- Import FAISS/LangChain en echec: verifier versions Python/dependances.
- Provider `mistral` sans cle: fallback auto HF avec warning.
- Index introuvable: rebuild via `scripts/build_index.py`.

### Etape 4

- `MISTRAL_API_KEY` manquante: generation impossible (retrieval local reste disponible).
- Quota/API Mistral: utiliser `scripts/evaluate_smoke.py --offline` pour test sans reseau.
- Reponse sans sources: la question est trop large ou le dataset/index est incomplet.

## Notebook de validation

Notebook pedagogique:

- `notebooks/validation_etapes_1_2.ipynb`

Il couvre les validations et demonstrations des etapes 1 a 3.

## Structure du depot (Etapes 1-4)

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
│   ├── eval/
│   │   └── smoke_eval.jsonl
│   └── sample/
│       └── events_sample.jsonl
├── logs/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── mistral/
│   └── __init__.py
├── notebooks/
│   └── validation_etapes_1_2.ipynb
├── scripts/
│   ├── ask_local.py
│   ├── build_dataset.py
│   ├── build_index.py
│   ├── check_env.py
│   ├── evaluate_smoke.py
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
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   └── schema.py
│   └── rag/
│       ├── __init__.py
│       ├── context.py
│       ├── llm.py
│       ├── prompts.py
│       ├── retriever.py
│       ├── service.py
│       └── types.py
└── tests/
    ├── conftest.py
    ├── test_cleaning.py
    ├── test_client.py
    ├── test_indexing_chunking.py
    ├── test_indexing_pipeline.py
    ├── test_rag_prompting.py
    ├── test_rag_retrieval.py
    ├── test_rag_service.py
    └── test_schema.py
```

## Reproductibilite

- Dependances figees dans `requirements.txt`.
- Aucun secret versionne.
- Sorties runtime ignorees (`data/`, `artifacts/faiss_index/`, `reports/`, `logs/`).
- Pipelines parametrables via `config.yaml`, `configs/indexing.yaml` et variables d'environnement.
