# Rapport - Réduction du nombre de fichiers (Projet 09)

## 1) Périmètre de l'audit
Audit réalisé sur l'ensemble du dépôt présent dans le workspace, sans modification fonctionnelle du code.

Sources utilisées pour le diagnostic:
- arborescence complète du projet
- fichiers versionnés (`git ls-files`)
- répartition par dossiers, extensions, tailles et volumétrie

## 2) État actuel (chiffré)

### Fichiers versionnés
- **63 fichiers versionnés** actuellement.
- Répartition principale:
  - `src/`: 27
  - `tests/`: 12
  - `scripts/`: 8
  - reste (config, docs, data sample/eval, artifacts docs): 16

### Fichiers présents dans le workspace (hors `.git` / `.venv`)
- Environ **106 fichiers** visibles côté code/projet.
- Le principal facteur de hausse est l'UI HTML5UP copiée dans `src/api/templates` (assets + images + webfonts + sass).

### Points de fragmentation observés
1. **Scripts CLI dispersés** (8 scripts) avec des responsabilités proches.
2. **API Flask très découpée** (app/config/deps/routes/errors/exceptions/index_manager/web/schemas).
3. **Beaucoup de fichiers UI vendor** (notamment `sass/` et multiples formats de fonts).
4. **Tests atomisés en 12 fichiers** (lisible, mais volumineux en nombre de fichiers).
5. **Fichiers sentinelles** (`.gitkeep`, `__init__.py`) nombreux mais faibles en impact (quelques fichiers).

---

## 3) Réductions possibles sans perte fonctionnelle majeure

## A. Quick wins (faible risque)

1. **Réduire les assets frontend vendoriés**
- Supprimer les sources `sass/` si non utilisées dans le cycle build.
- Conserver uniquement les formats de fonts nécessaires (souvent `woff2` + `woff`).
- Garder `main.css`, `main.js`, 3 images de fond.

Impact estimé:
- **-20 à -28 fichiers** (selon formats conservés).
- Très faible impact runtime si sélection prudente.

2. **Nettoyage des fichiers parasites non utiles au projet**
- Éviter de conserver `.DS_Store` dans l'arborescence de travail.

Impact estimé:
- faible, mais hygiène importante.

3. **Regrouper les docs d'artifacts**
- `artifacts/README.md` + `artifacts/faiss_index/README.md` -> un seul README clair si acceptable.

Impact estimé:
- **-1 fichier**.

---

## B. Rationalisation modérée (risque faible à moyen)

1. **Fusionner les scripts CLI dans un seul point d'entrée**
- Conserver un `scripts/cli.py` (ou `scripts/project.py`) avec sous-commandes:
  - `check-env`, `build-dataset`, `build-index`, `query-index`, `ask-local`, `evaluate`, `run-api`, `api-test`, `bootstrap`.
- Garder éventuellement 1 ou 2 wrappers de compatibilité.

Impact estimé:
- de 8 scripts à 1-3 scripts.
- **-5 à -7 fichiers**.

2. **Regrouper certains modules API Flask**
- Candidats de fusion naturelle:
  - `errors.py` + `exceptions.py`
  - `config.py` + partie settings dans `app.py`
- Laisser `routes.py` et `deps.py` séparés (utile pour testabilité).

Impact estimé:
- **-2 à -3 fichiers**.

3. **Regrouper les tests par domaine**
- API: 4 fichiers -> 2 fichiers.
- RAG: 3 fichiers -> 1-2 fichiers.
- Indexing: 2 fichiers -> 1 fichier.

Impact estimé:
- **-4 à -6 fichiers**.

---

## C. Rationalisation agressive (risque moyen)

1. **Fusion de modules métier par couche**
- `src/rag/`: 6 fichiers -> 3 fichiers (ex: `core.py`, `llm.py`, `types.py`).
- `src/indexing/`: 5 fichiers -> 2-3 fichiers.
- `src/preprocess/`: 2 fichiers -> 1 fichier.

Impact estimé:
- **-6 à -9 fichiers**.

Risques:
- baisse de lisibilité pédagogique (important pour soutenance)
- conflits de merge plus fréquents
- augmentation de la taille des fichiers (>400-600 lignes)

---

## 4) Scénarios cibles

## Scénario 1 - Prudent (recommandé)
Objectif: simplifier sans dégrader la lisibilité du projet.

Actions:
- Quick wins assets frontend
- fusion légère des scripts
- fusion minimale API (`errors`/`exceptions`)

Gain attendu:
- **-25 à -35 fichiers** environ (si assets optimisés fortement).

## Scénario 2 - Équilibré
Actions scénario 1 + regroupement tests par domaine.

Gain attendu:
- **-30 à -42 fichiers**.

## Scénario 3 - Compact maximal
Actions scénario 2 + fusion modules métier (RAG/Indexing/Preprocess).

Gain attendu:
- **-38 à -50 fichiers**.
- Risque plus élevé sur maintenabilité et qualité pédagogique.

---

## 5) Recommandation
Recommandation: **Scénario 1** dans un premier temps.

Pourquoi:
- meilleur ratio gain/risque
- conserve l'architecture didactique utile pour l'évaluation
- réduit surtout les fichiers "vendor" et la dispersion des scripts, qui sont les plus gros contributeurs au nombre de fichiers

---

## 6) Plan d'exécution proposé (quand vous donnerez le go)

1. Standardiser la stratégie frontend:
- soit assets locaux minimaux,
- soit externalisation CDN (avec fallback local si besoin).

2. Introduire un CLI unique (`scripts/cli.py`) avec compatibilité temporaire.

3. Fusionner `errors.py` et `exceptions.py` côté API.

4. Regrouper tests par domaine (sans réduire la couverture).

5. Vérifier la non-régression:
- `pytest -q`
- `python scripts/check_env.py`
- smoke API (`/health`, `/metadata`, `/ask`, `/rebuild`)

