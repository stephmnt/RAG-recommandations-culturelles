# POC RAG Puls-Events (Etapes 1 a 6)

Ce depot couvre:

- Etape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
- Etape 2: recuperation + nettoyage OpenAgenda pour produire un dataset pret a indexer
- Etape 3: chunking + embeddings + indexation FAISS persistante
- Etape 4: moteur RAG (retrieval + generation Mistral)
- Etape 5: API REST Flask (`/ask`, `/rebuild`, `/health`, `/metadata`)
- Etape 6: conteneurisation Docker + run local pour la demo

## Choix techniques

- Gestion d'environnement: `requirements.txt` avec versions fixes.
- Portabilite: `faiss-cpu`.
- Secrets: `.env` local uniquement, jamais versionne.
- Zone geographique cible: `Departement de l'Herault (34)`.
- Fenetre temporelle: historique glissant 365 jours + horizon a venir configurable.
- Sortie etape 2: `events_processed.parquet` contenant `document_text` et `retrieval_metadata`.
- Sortie etape 3: index FAISS persiste dans `artifacts/faiss_index`.
- Etape 5/6: API Flask avec cache service RAG et endpoint admin de rebuild.

## Prerequis

- Python 3.10 ou 3.11 recommande
- pip recent
- Docker Desktop (pour etape 6)

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

## Configuration (`.env`)

1. Copier l'exemple:

```bash
cp .env.example .env
```

2. Renseigner au minimum:

- `OPENAGENDA_API_KEY` (etape 2)
- `MISTRAL_API_KEY` (etapes 4-6)
- `ADMIN_TOKEN` (protection endpoint `/rebuild`)

3. Ne jamais commiter `.env`.

## Smoke test environnement (Etape 1)

```bash
python3 scripts/check_env.py
```

Le script affiche les versions et verifie les imports critiques.

## Construire les donnees (Etape 2)

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
  --config configs/indexing.yaml \
  --input data/processed/events_processed.parquet \
  --output artifacts/faiss_index
```

Sorties:

- `artifacts/faiss_index/index.faiss`
- `artifacts/faiss_index/index.pkl`
- `artifacts/faiss_index/index_metadata.json`

## Tester le moteur RAG en local (Etape 4)

```bash
python3 scripts/ask_local.py --query "Quels evenements jazz a Montpellier cette semaine ?" --debug
```

## API Flask (Etape 5)

### Lancer l'API

```bash
python3 scripts/run_api.py --host 127.0.0.1 --port 8000
```

### Endpoints

- `GET /health`: etat de l'API et presence index
- `GET /metadata`: metadata index + config RAG
- `POST /ask`: question utilisateur + top_k optionnel + filtres optionnels
- `POST /rebuild`: `mode=reload|rebuild`, protege par header `X-ADMIN-TOKEN`

### Exemple `/ask`

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels evenements jazz dans l Herault ?",
    "top_k": 6,
    "debug": true
  }' | jq
```

### Exemple `/rebuild` (reload)

```bash
curl -s -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -H "X-ADMIN-TOKEN: ${ADMIN_TOKEN}" \
  -d '{"mode":"reload"}' | jq
```

### Smoke test API

```bash
python3 scripts/api_test.py --base-url http://127.0.0.1:8000 --admin-token "$ADMIN_TOKEN"
```

## Conteneurisation Docker (Etape 6)

### Build image

```bash
docker build -t puls-events-rag:step6 .
```

### Run container

```bash
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  puls-events-rag:step6
```

### Run via Docker Compose

```bash
docker compose up --build
```

### Script demo etape 6 (build index + build image + run)

```bash
./scripts/step6_docker_demo.sh
```

### Scenarios de demo proposes (soutenance)

1. `Quels evenements jazz dans l Herault cette semaine ?`
2. `Je cherche une sortie famille a Montpellier ce week-end.`
3. `Quelles expositions autour de Beziers dans les 30 prochains jours ?`

## Commande end-to-end locale

Le bootstrap existant execute l'ordre complet:

```bash
python3 scripts/bootstrap_app.py
```

Etapes executees:

1. `scripts/check_env.py`
2. `scripts/build_dataset.py` (si dataset manquant/invalide)
3. `scripts/build_index.py`
4. `scripts/run_api.py`
5. `scripts/api_test.py`

## Tests

```bash
pytest -q
```

## Depannage rapide

- `INDEX_UNAVAILABLE`: lancer `scripts/build_index.py` ou `/rebuild` mode `rebuild`.
- `UNAUTHORIZED` sur `/rebuild`: header `X-ADMIN-TOKEN` absent.
- `FORBIDDEN` sur `/rebuild`: token invalide.
- `MISTRAL_API_KEY` manquante: l'API repondra avec fallback sur la generation.
- OpenAgenda `403`: verifier la cle et les droits de lecture.

## Arborescence (principale)

```text
.
├── Dockerfile
├── docker-compose.yml
├── config.yaml
├── configs/
│   └── indexing.yaml
├── requirements.txt
├── scripts/
│   ├── api_test.py
│   ├── ask_local.py
│   ├── bootstrap_app.py
│   ├── build_dataset.py
│   ├── build_index.py
│   ├── check_env.py
│   ├── run_api.py
│   └── step6_docker_demo.sh
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── config.py
│   │   ├── deps.py
│   │   ├── errors.py
│   │   ├── exceptions.py
│   │   ├── index_manager.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── indexing/
│   ├── openagenda/
│   ├── preprocess/
│   └── rag/
└── tests/
    ├── test_api_ask.py
    ├── test_api_health_metadata.py
    ├── test_api_rebuild.py
    └── ...
```
