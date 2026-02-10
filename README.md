# POC RAG Puls-Events (Étapes 1 et 2)

Ce dépôt couvre uniquement :

* Étape 1: environnement reproductible (LangChain + Mistral + FAISS CPU)
* Étape 2: récupération + nettoyage OpenAgenda pour produire un dataset prêt à indexer

Le scope volontairement exclu pour l'instant :

* index FAISS final
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

## Structure du dépôt (Étapes 1-2)

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

## Reproductibilité

* Dépendances figées dans `requirements.txt`.
* Aucun secret versionné.
* Aucun fichier généré versionné (hors échantillon `data/sample/`).
* Pipeline paramétrable via `config.yaml` + variables d'environnement.
