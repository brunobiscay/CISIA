# Diagnostic assisté et tri d'urgence multimodal en télémédecine

Système d'intelligence artificielle de tri médical : à partir de données hybrides (constantes
vitales, antécédents, description libre des symptômes), le modèle prédit un **niveau d'urgence**
à 3 classes (0 : non urgent, 1 : urgence relative, 2 : urgence vitale), pour aider à prioriser la
prise en charge dans un contexte d'engorgement des services d'urgence et de développement de la
télémédecine.

Projet réalisé dans le cadre de la certification *Promo Upskilling Atlas CISIA* (Session-00301513).

## Sommaire

- [Contexte et enjeux](#contexte-et-enjeux)
- [Modèle retenu](#modèle-retenu)
- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Démarrage rapide](#démarrage-rapide)
- [Pipeline MLOps](#pipeline-mlops)
- [Tests](#tests)
- [Documentation](#documentation)

## Contexte et enjeux

Le jeu de données source (2000 échantillons) combine données tabulaires (âge, sexe, constantes
vitales, antécédents, durée des symptômes) et texte libre (plainte du patient ou compte-rendu du
régulateur). L'enjeu métier central : **une erreur de sous-triage** (classer en urgence vitale une
situation réellement non urgente est dommageable, mais l'inverse — classer en non urgent une
urgence vitale — est une erreur bien plus grave). Le modèle a donc été optimisé pour minimiser ce
taux d'erreurs critiques (`critical_undertriage_rate`), pas seulement l'accuracy globale.

Les données personnelles ont été traitées dans le respect du RGPD (anonymisation/pseudonymisation,
modélisation MERISE, suppression des champs sensibles non justifiés — voir le rapport pour le
détail de cette analyse).

## Modèle retenu

`RF_IMPORTANCE_20 + LGBMClassifier` (sélection des 20 features les plus importantes via Random
Forest, puis classification par LightGBM), hyperparamètres optimisés via Optuna :

| Métrique | Valeur |
|---|---|
| F1-macro | 0.948 |
| Accuracy | 0.958 |
| Taux d'erreurs critiques (`critical_undertriage_rate`) | 4.4 % |

Plusieurs architectures (régression logistique, Random Forest, LightGBM, réseau de neurones,
régression ordinale) et jeux de features (toutes les variables, sélection `SelectKBest`, sélection
par importance Random Forest) ont été comparés ; le détail de cette comparaison est dans
`data/artifacts/summary.csv` et `data/artifacts/tuning_summary.csv`, et dans le rapport.

## Architecture

```
                         ┌─────────────┐
                         │  Streamlit  │  (frontend, formulaire + feedback)
                         └──────┬──────┘
                                │ HTTP (X-API-Key)
                         ┌──────▼──────┐        ┌─────────────┐
                         │   FastAPI   │◄───────►│   MLflow    │  (Model Registry,
                         │  (backend)  │         │  (tracking) │   tracking server)
                         └──────┬──────┘        └──────▲──────┘
                                │                       │
                         ┌──────▼──────┐                │
                         │  PostgreSQL │◄───────────────┘
                         │  (données + │
                         │   MLflow +  │        ┌─────────────┐
                         │   drift)    │◄───────►│   Prefect   │  (drift planifié,
                         └─────────────┘         │ (orchestr.) │   réentraînement,
                                                  └─────────────┘   promotion conditionnelle)
```

Tous les services sont conteneurisés (Docker Compose en local, images publiées sur DockerHub via
CI/CD). Le jeu de données d'entraînement est versionné avec DVC.

## Structure du projet

```
sources/         Import des données brutes (CSV) en base PostgreSQL
common/          Code partagé : schéma SQLAlchemy, schéma de features, calcul d'embedding CamemBERT
ml/              Pipeline ML : extraction, feature engineering, entraînement, tuning Optuna,
                 monitoring de dérive (Evidently), préparation des données de production,
                 promotion de modèle dans le Model Registry
backend/         API FastAPI (/predict, /predict/{id}/feedback) + tests
frontend/        Interface Streamlit (saisie, affichage du résultat, feedback)
mlflow/          Image du serveur de tracking MLflow (backend-store Postgres)
orchestration/   Flows Prefect (drift_check_flow, retrain_flow) + déploiements planifiés
data/            Jeu de données (versionné via DVC) et artefacts d'entraînement
.github/         Workflows CI (lint, tests, build) et CD (push DockerHub)
```

## Démarrage rapide

### Avec Docker (recommandé)

```bash
cp .env.example .env   # renseigner les secrets (Postgres, API_KEY, ...)
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend Streamlit | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4200 |

Une version `docker-compose.prod.yml`, utilisant les images publiées sur DockerHub
(`brunobiscay/triage-*`) plutôt qu'un build local, est disponible pour un déploiement client.

### En local (développement du pipeline ML)

```bash
uv sync
uv run python -m ml.extract_dataset   # construit data/dataset.parquet depuis la base
uv run python -m ml.train             # entraîne toutes les combinaisons feature_set x modèle
uv run python -m ml.tune              # tuning Optuna des meilleurs candidats
```

## Pipeline MLOps

Le projet va au-delà de l'entraînement d'un modèle : c'est une chaîne complète mise en production
et maintenue automatiquement, en 5 volets :

1. **Serving** — Model Registry MLflow (stage `Production`), API `/predict`, frontend Streamlit,
   boucle de feedback différé (`/predict/{id}/feedback`) pour collecter les vraies décisions
   médicales a posteriori.
2. **Monitoring** — détection de dérive des données et de performance (Evidently AI), persistée et
   comparée à un seuil d'alerte.
3. **Orchestration** — Prefect : vérification de dérive planifiée, déclenchant un réentraînement
   automatique en cas d'alerte (+ filet de sécurité mensuel), avec **promotion conditionnelle** du
   nouveau modèle (uniquement s'il n'est pas moins bon que le modèle en production).
4. **CI/CD** — GitHub Actions : lint, tests, build de validation sur chaque PR ; build + push des 5
   images vers DockerHub sur `main`.
5. **Versioning des données** — DVC, avec intégration automatique dans le flow de réentraînement.

## Tests

```bash
uv run ruff check .
uv run pytest backend/tests/ -v
```

## Documentation

Le détail des choix techniques (architecture de données RGPD/MERISE, comparaison des modèles,
scénarios d'analyse de sensibilité, justification des choix d'industrialisation) est disponible
dans le rapport de certification (`PlaybookPDF_BrunoBiscay_Cisia-00301513.pdf`).
