# 📚 SAÉ 2.04 - Projet de Base de Données DUT Informatique

## 🎓 Université Rennes - IUT de Lannion  
**Semestre 2 – DUT Informatique**  
**Binôme :** _kXX_ (à adapter avec votre numéro de binôme)

---

## 📌 Objectif du projet

Ce projet a pour but de **concevoir, peupler et exploiter** une base de données réaliste basée sur les admissions, inscriptions et résultats des étudiants en DUT Informatique. Il est découpé en **trois grandes parties** :

1. **Conception du schéma relationnel**
2. **Peuplement de la base avec des fichiers CSV**
3. **Exploitation statistique via Python**

---

## 🧱 Partie 1 — Conception de la Base de Données

### ✅ Objectif
Traduction d’un diagramme de classes UML en un schéma relationnel PostgreSQL.

### 📄 Contenu livré
- `sae204_kXX_partie1.sql` : Script SQL complet de création de la base (schéma nommé `partie1`)
- Contraintes d’intégrité référentielles nommées explicitement
- Préfixe `_` appliqué aux noms de tables (_etudiant, _semestre, etc.)

---

## 🗂 Partie 2 — Peuplement de la Base

### ✅ Objectif
Utilisation des fichiers extraits de Parcoursup, Apogée, etc. pour remplir les tables de la base tout en corrigeant les incohérences ou données manquantes.

### 📄 Contenu livré
- `sae204_kXX_partie2.sql` : Script SQL de peuplement à partir du fichier `sae204_partie2_schema.sql`
- Import des données via `WbImport` ou autres moyens
- Gestion des valeurs spéciales (e.g., `~`, `-c-`) dans les notes
- Adaptation du schéma aux données réelles (relaxation des contraintes si nécessaire)
- Réflexion sur l’efficacité computationnelle de l’import (commentée dans le script)

---

## 📊 Partie 3 — Analyse Statistique

### ✅ Objectif
Réaliser une analyse statistique via une vue extraite de la base (≥ 5 variables, ≥ 20 lignes), corrélations et régression linéaire multiple en Python.

### 📄 Contenu livré
- `rapport_statistique.pdf` : Rapport structuré comprenant :
  - Définition d’une problématique
  - Description de la vue
  - Visualisations graphiques
  - Matrice de corrélation
  - Régression linéaire multiple
  - Interprétations et conclusion

- `vue_analyse.csv` : Vue extraite de la base
- `analyse.py` : Code Python complet avec :
  - Suffixes normalisés : `Ar`, `DF`, `Li`, `N`
  - Commentaires et captures intégrées dans le PDF

---

## 📁 Structure du dépôt

```bash
.
├── partie1/
│   └── sae204_kXX_partie1.sql
├── partie2/
│   └── sae204_kXX_partie2.sql
├── statistique/
│   ├── rapport_statistique.pdf
│   ├── vue_analyse.csv
│   └── analyse.py
├── Sujet.pdf
├── sae204_2025_partie2.pdf
└── README.md
