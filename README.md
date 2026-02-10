# POC RAG Puls-Events (Etapes 1 a 5)

Ce depot couvre:

- Etape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
- Etape 2: recuperation + nettoyage OpenAgenda
- Etape 3: chunking + embeddings + indexation FAISS persistante
- Etape 4: moteur RAG (retrieval + generation Mistral)
- Etape 5: API REST Flask locale (`/ask`, `/rebuild`, `/health`, `/metadata`)

Hors scope actuel:

- Docker (etape 6)
- evaluation RAGAS complete

## Choix techniques

- Environnement: `requirements.txt` fige.
- Portabilite: `faiss-cpu`.
- Secrets: `.env` local, jamais versionne.
- Zone metier: departement de l'Herault (34).
- Donnees: `document_text` + `retrieval_metadata` prepares en etape 2.
- Retrieval: FAISS + LangChain avec imports compatibles `langchain_community`/`langchain`.
- API Flask:
  - validation des payloads,
  - erreurs JSON standardisees,
  - token admin pour `/rebuild`,
  - cache memoire du service/index,
  - logs avec `request_id` et latence.

## Prerequis

- Python 3.10 ou 3.11 recommande
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

## Configuration (.env)

1. Copier l'exemple:

```bash
cp .env.example .env
```

2. Renseigner au minimum:

- `OPENAGENDA_API_KEY`
- `MISTRAL_API_KEY` (obligatoire en mode live RAG)
- `ADMIN_TOKEN` (obligatoire pour `/rebuild`)

### Variables d'environnement API (Etape 5)

| Variable | Defaut | Description |
|---|---|---|
| `FLASK_ENV` | `dev` | Mode Flask |
| `HOST` | `127.0.0.1` | Host API |
| `PORT` | `8000` | Port API |
| `LOG_LEVEL` | `INFO` | Niveau de logs |
| `MISTRAL_API_KEY` | `""` | Cle Mistral |
| `MISTRAL_MODEL` | `mistral-small-latest` | Modele generation |
| `INDEX_PATH` | `artifacts/faiss_index` | Dossier index FAISS |
| `DATASET_PATH` | `data/processed/events_processed.parquet` | Dataset source rebuild |
| `INDEXING_CONFIG_PATH` | `configs/indexing.yaml` | Config indexation |
| `EMBEDDING_PROVIDER` | `huggingface` | Provider embeddings |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modele embeddings |
| `ADMIN_TOKEN` | `""` | Token header `X-ADMIN-TOKEN` |
| `MAX_TOP_K` | `10` | Limite de securite top_k |
| `DEFAULT_TOP_K` | `6` | top_k par defaut |
| `MAX_QUESTION_CHARS` | `500` | Taille max question |
| `CONTEXT_MAX_CHARS` | `8000` | Taille max contexte RAG |
| `MAX_SOURCES` | `5` | Max sources retournees |
| `PROMPT_VERSION` | `v1` | Version prompt |

## Smoke test environnement (Etape 1)

```bash
python3 scripts/check_env.py
```

## Etape 2 - Construire le dataset

```bash
python3 scripts/build_dataset.py --config config.yaml
```

Sorties:

- `data/raw/events_raw.jsonl`
- `data/processed/events_processed.parquet`
- `logs/build_dataset.log`

## Etape 3 - Construire l'index FAISS

```bash
python3 scripts/build_index.py \
  --input data/processed/events_processed.parquet \
  --output artifacts/faiss_index
```

Test recherche locale:

```bash
python3 scripts/query_index.py --query "concert jazz montpellier" --k 5
```

## Etape 4 - Moteur RAG local

Question locale (sans API):

```bash
python3 scripts/ask_local.py --query "Quels evenements jazz dans l'Herault ?" --debug
```

Smoke eval simple:

```bash
python3 scripts/evaluate_smoke.py --offline
```

## API Flask (Etape 5)

### Lancer l'API

```bash
python3 scripts/run_api.py
```

Options utiles:

```bash
python3 scripts/run_api.py --host 127.0.0.1 --port 8000 --log-level INFO
```

### Endpoints

#### `POST /ask`

Request JSON:

```json
{
  "question": "Quels concerts jazz dans l'Herault cette semaine ?",
  "top_k": 6,
  "debug": false,
  "filters": {
    "city": "Montpellier",
    "date_from": "2026-02-01",
    "date_to": "2026-02-28"
  }
}
```

- `question` obligatoire non vide.
- `top_k` est borne entre `1` et `MAX_TOP_K`.
- `filters` est applique en post-filter sur les sources (limite documentee).

Response `200`:

```json
{
  "question": "...",
  "answer": "...",
  "sources": [...],
  "meta": {...}
}
```

#### `POST /rebuild`

Header obligatoire:

- `X-ADMIN-TOKEN: <ADMIN_TOKEN>`

Request JSON:

```json
{
  "mode": "rebuild",
  "dataset_path": "data/processed/events_processed.parquet",
  "index_path": "artifacts/faiss_index"
}
```

- `mode = reload`: recharge l'index depuis disque.
- `mode = rebuild`: reconstruit l'index complet puis le recharge.

Concurrence:

- un seul rebuild a la fois (mutex process).
- pendant rebuild, l'API continue de servir l'ancien index en cache si disponible.

#### `GET /health`

Response `200`:

```json
{
  "status": "ok",
  "api": "up",
  "index_loaded": true,
  "mistral_configured": true,
  "version": "0.1.0",
  "timestamp": "..."
}
```

#### `GET /metadata`

Response `200`:

```json
{
  "index": {
    "path": "...",
    "build_date": "...",
    "num_events": 123,
    "num_chunks": 456,
    "embedding_model": "...",
    "dataset_hash": "..."
  },
  "rag": {
    "default_top_k": 6,
    "max_top_k": 10,
    "prompt_version": "v1",
    "llm_model": "mistral-small-latest"
  }
}
```

### Exemples curl

```bash
curl -s http://127.0.0.1:8000/health | jq
```

```bash
curl -s http://127.0.0.1:8000/metadata | jq
```

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels evenements jazz dans l'"'"'Herault ?","top_k":6}' | jq
```

```bash
curl -s -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -H "X-ADMIN-TOKEN: ${ADMIN_TOKEN}" \
  -d '{"mode":"reload"}' | jq
```

### Exemple Python requests

```python
import requests

base = "http://127.0.0.1:8000"
print(requests.get(f"{base}/health", timeout=10).json())
print(requests.post(
    f"{base}/ask",
    json={"question": "Quels concerts jazz dans l'Herault ?", "top_k": 6},
    timeout=30,
).json())
```

### Script de smoke manuel API

```bash
python3 scripts/api_test.py --base-url http://127.0.0.1:8000
```

Mode sans appel live generation:

```bash
python3 scripts/api_test.py --offline
```

## Codes d'erreur API

- `400` `INVALID_REQUEST`: payload invalide, question vide, mode invalide.
- `401` `MISSING_ADMIN_TOKEN`: header admin absent.
- `403` `INVALID_ADMIN_TOKEN`: token admin incorrect.
- `422` `INVALID_SCHEMA`: schema JSON invalide.
- `503` `INDEX_UNAVAILABLE` / `REBUILD_IN_PROGRESS` / `ADMIN_TOKEN_NOT_CONFIGURED`.
- `500` `INTERNAL_ERROR`.

Format standard:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "question is required",
    "details": {},
    "request_id": "..."
  }
}
```

## Tests

Lancer tous les tests unitaires + API:

```bash
pytest -q
```

Les tests API utilisent `Flask test_client` et des mocks (pas de dependance reseau).

## Troubleshooting

- `INDEX_UNAVAILABLE`:
  - construire l'index (`scripts/build_index.py`) ou appeler `/rebuild` mode `rebuild`.
- `/rebuild` retourne `401/403`:
  - verifier `ADMIN_TOKEN` et header `X-ADMIN-TOKEN`.
- Mistral non configure:
  - verifier `MISTRAL_API_KEY`.
- Erreurs imports LangChain/FAISS:
  - verifier `pip install -r requirements.txt` et version Python.

## Structure du depot (Etapes 1-5)

```text
.
├── .env.example
├── README.md
├── requirements.txt
├── config.yaml
├── configs/
│   └── indexing.yaml
├── scripts/
│   ├── api_test.py
│   ├── run_api.py
│   ├── build_dataset.py
│   ├── build_index.py
│   ├── check_env.py
│   ├── query_index.py
│   ├── ask_local.py
│   └── evaluate_smoke.py
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── config.py
│   │   ├── deps.py
│   │   ├── errors.py
│   │   ├── index_manager.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── indexing/
│   ├── openagenda/
│   ├── preprocess/
│   └── rag/
└── tests/
    ├── test_api_ask.py
    ├── test_api_rebuild.py
    ├── test_api_health_metadata.py
    └── ...
```
