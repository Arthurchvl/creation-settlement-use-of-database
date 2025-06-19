# ANALYSE STATISTIQUE DES MOYENNES SEMESTRIELLES - RÉGRESSION LINÉAIRE MULTIPLE
# =============================================================================
# Ce script analyse les données de moyennes d'étudiants sur 4 semestres
# Il effectue une régression linéaire pour prédire la moyenne du semestre 4
# à partir des moyennes des semestres 1, 2 et 3

# On importe les bibliothèques nécessaires :
# - pandas pour manipuler les données sous forme de tableau (DataFrame)
# - numpy pour faire des calculs mathématiques et statistiques
# - matplotlib.pyplot pour faire des graphiques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# FONCTIONS POUR LA RÉGRESSION LINÉAIRE
# ----------------------------
# Ces fonctions permettent de calculer une régression linéaire "à la main"
# sans utiliser les bibliothèques comme scikit-learn


def calculer_coefficients_regression_lineaire(variables_explicatives, variable_cible):
    """
    Calcule les coefficients (theta) d'un modèle de régression linéaire multiple.
    
    EXPLICATION SIMPLE :
    - On veut trouver une équation du type : S4 = a0 + a1*S1 + a2*S2 + a3*S3
    - Cette fonction calcule les coefficients a0, a1, a2, a3
    - Elle utilise la méthode des "moindres carrés" (formule mathématique)
    
    PARAMÈTRES :
    - variables_explicatives : matrice (tableau) contenant les variables explicatives (S1, S2, S3)
    - variable_cible : vecteur contenant la variable à prédire (S4)
    
    RETOUR :
    - Un tableau avec les coefficients [a0, a1, a2, a3]
    """
    # Nombre d'étudiants dans notre échantillon
    nombre_echantillons = variables_explicatives.shape[0]  # .shape[0] donne le nombre de lignes
    
    # On ajoute une colonne de 1 à gauche de X pour calculer l'intercept (a0)
    # C'est une astuce mathématique pour inclure le terme constant dans l'équation
    variables_explicatives_augmentees = np.hstack((np.ones((nombre_echantillons, 1)), variables_explicatives))
    
    # Application de la formule des moindres carrés : θ = (X^T * X)^(-1) * X^T * y
    # Ne vous inquiétez pas des détails mathématiques, c'est la formule standard
    coefficients_theta = np.linalg.inv(variables_explicatives_augmentees.T @ variables_explicatives_augmentees) @ variables_explicatives_augmentees.T @ variable_cible
    
    # On transforme le résultat en tableau à une dimension pour simplifier
    return coefficients_theta.flatten()


def predire_variable_cible(variables_explicatives, coefficients_theta):
    """
    Prédit les valeurs de y à partir de X et des coefficients theta.
    
    EXPLICATION SIMPLE :
    - Une fois qu'on a nos coefficients, on peut prédire S4 pour n'importe quel étudiant
    - On applique l'équation : S4_prédit = a0 + a1*S1 + a2*S2 + a3*S3
    
    PARAMÈTRES :
    - variables_explicatives : données des étudiants (S1, S2, S3)
    - coefficients_theta : coefficients calculés précédemment
    
    RETOUR :
    - Les prédictions de S4 pour chaque étudiant
    """
    nombre_echantillons = variables_explicatives.shape[0]
    # On ajoute la colonne de 1 comme dans la fonction précédente
    variables_explicatives_augmentees = np.hstack((np.ones((nombre_echantillons, 1)), variables_explicatives))
    # On calcule les prédictions en multipliant les données par les coefficients
    predictions_variable_cible = variables_explicatives_augmentees @ coefficients_theta  # @ est l'opérateur de multiplication matricielle
    return predictions_variable_cible


def calculer_coefficient_correlation_multiple(valeurs_reelles, valeurs_predites):
    """
    Calcule le coefficient de détermination R² (qualité du modèle).
    
    EXPLICATION SIMPLE :
    - R² mesure la qualité de notre prédiction sur une échelle de 0 à 1
    - R² = 0 : notre modèle ne prédit rien (très mauvais)
    - R² = 1 : notre modèle prédit parfaitement (excellent)
    - R² = 0.8 : notre modèle explique 80% de la variabilité (très bon)
    
    PARAMÈTRES :
    - valeurs_reelles : vraies valeurs de S4
    - valeurs_predites : valeurs prédites par notre modèle
    
    RETOUR :
    - La valeur de R² entre 0 et 1
    """
    # On s'assure que nos données sont sous forme de tableaux simples
    valeurs_reelles = np.ravel(valeurs_reelles)  # aplatissement des tableaux
    valeurs_predites = np.ravel(valeurs_predites)
    
    # Calcul de la somme des carrés des erreurs de prédiction
    somme_carres_residus = np.sum((valeurs_reelles - valeurs_predites)**2)  # somme des carrés des résidus
    
    # Calcul de la variance totale des vraies valeurs
    somme_carres_totale = np.sum((valeurs_reelles - np.mean(valeurs_reelles))**2)  # somme totale des carrés
    
    # Formule du R² : R² = 1 - (erreur du modèle / variance totale)
    coefficient_determination = 1 - somme_carres_residus / somme_carres_totale
    return coefficient_determination


def normaliser_donnees(donnees_brutes):
    """
    Normalise les données : centre (moyenne 0) et réduit (écart-type 1)
    
    EXPLICATION SIMPLE :
    - Parfois les variables ont des échelles très différentes (ex: notes sur 20 vs âge en années)
    - La normalisation met toutes les variables sur la même échelle
    - Chaque variable aura une moyenne de 0 et un écart-type de 1
    - Cela améliore la stabilité des calculs et permet de comparer les coefficients
    
    FORMULE : (valeur - moyenne) / écart-type
    
    PARAMÈTRES :
    - donnees_brutes : données à normaliser
    
    RETOUR :
    - donnees_normalisees : données normalisées
    - moyennes_originales : moyennes originales (pour pouvoir "dénormaliser" plus tard)
    - ecarts_types_originaux : écarts-types originaux
    """
    moyennes_originales = np.mean(donnees_brutes, axis=0)  # moyenne de chaque colonne (variable)
    ecarts_types_originaux = np.std(donnees_brutes, axis=0)   # écart-type de chaque colonne
    donnees_normalisees = (donnees_brutes - moyennes_originales) / ecarts_types_originaux    # transformation de normalisation
    return donnees_normalisees, moyennes_originales, ecarts_types_originaux

# ----------------------------
# 1. IMPORT ET EXPLORATION DES DONNÉES
# ----------------------------
# Dans cette section, on charge les données et on fait une première exploration

print("=== 1. IMPORT ET EXPLORATION DES DONNÉES ===")

# Chargement du fichier CSV dans un DataFrame (tableau pandas)
# Un DataFrame est comme un tableau Excel en Python
donnees_etudiants_df = pd.read_csv('Vue.csv')

# Affichage des informations de base sur notre dataset
print("Shape du DataFrame:", donnees_etudiants_df.shape)  # (nombre de lignes, nombre de colonnes)
print("\nPremières lignes:")
print(donnees_etudiants_df.head())  # Affiche les 5 premières lignes pour voir à quoi ressemblent les données

print("\nInformations sur le dataset:")
print(donnees_etudiants_df.info())  # Affiche les types de colonnes et les valeurs manquantes

print("\nStatistiques descriptives:")
print(donnees_etudiants_df.describe())  # Statistiques de base : moyenne, min, max, écart-type, etc.

# Vérification des valeurs manquantes dans la colonne département
# Les valeurs manquantes (NaN) peuvent poser des problèmes dans les calculs
print(f"\nValeurs manquantes dans dept_etablissement: {donnees_etudiants_df['dept_etablissement'].isna().sum()}")

# ----------------------------
# 2. ANALYSE DES CORRÉLATIONS
# ----------------------------
# Dans cette section, on analyse les relations entre les moyennes des différents semestres

print("\n=== 2. ANALYSE DES CORRÉLATIONS ===")

# On définit une liste avec les noms des colonnes contenant les moyennes
colonnes_moyennes_semestrielles = ['moyenne_semestre_1', 'moyenne_semestre_2',
                                   'moyenne_semestre_3', 'moyenne_semestre_4']

# Calcul de la matrice de corrélation entre toutes les moyennes semestrielles
# La corrélation mesure la relation linéaire entre deux variables (-1 à +1)
# +1 = relation parfaitement positive, 0 = pas de relation, -1 = relation négative
matrice_correlation_moyennes_df = donnees_etudiants_df[colonnes_moyennes_semestrielles].corr()

print("Matrice de corrélation:")
print(matrice_correlation_moyennes_df)

# Création d'un graphique visuel de la matrice de corrélation
# Les couleurs chaudes (rouge) = corrélation positive forte
# Les couleurs froides (bleu) = corrélation négative forte
plt.figure(figsize=(10, 8))  # Taille du graphique
image_matrice_correlation = plt.imshow(matrice_correlation_moyennes_df, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Affichage des valeurs numériques dans chaque case de la matrice
for indice_ligne in range(len(matrice_correlation_moyennes_df.columns)):
    for indice_colonne in range(len(matrice_correlation_moyennes_df.columns)):
        # On écrit la valeur de corrélation avec 3 décimales dans chaque case
        plt.text(indice_colonne, indice_ligne, f'{matrice_correlation_moyennes_df.iloc[indice_ligne, indice_colonne]:.3f}',
                 ha='center', va='center', fontsize=10)

# Configuration des étiquettes et du titre
plt.xticks(range(len(matrice_correlation_moyennes_df.columns)), matrice_correlation_moyennes_df.columns, rotation=45)
plt.yticks(range(len(matrice_correlation_moyennes_df.columns)), matrice_correlation_moyennes_df.columns)
plt.colorbar(image_matrice_correlation, label='Coefficient de corrélation')  # Légende des couleurs
plt.title('Matrice de corrélation entre les moyennes semestrielles')
plt.tight_layout()  # Ajuste automatiquement l'espacement
plt.show()  # Affiche le graphique

# ----------------------------
# 3. CORRÉLATIONS AVEC LE SEMESTRE 4
# ----------------------------
# Ici on se concentre spécifiquement sur les corrélations avec S4 (notre variable cible)

print("\n=== 3. CORRÉLATIONS AVEC LA MOYENNE DU SEMESTRE 4 ===")

# On extrait les corrélations avec S4 et on les trie par ordre décroissant
# Cela nous dit quels semestres sont les plus liés à S4
correlations_avec_semestre_4 = donnees_etudiants_df[colonnes_moyennes_semestrielles].corr()['moyenne_semestre_4'].sort_values(ascending=False)

print("Corrélations avec la moyenne du semestre 4:")
# On affiche toutes les corrélations sauf celle de S4 avec lui-même (qui vaut 1)
for nom_variable, valeur_correlation in correlations_avec_semestre_4.items():
    if nom_variable != 'moyenne_semestre_4':
        print(f"{nom_variable}: {valeur_correlation:.3f}")

# ----------------------------
# 4. RÉGRESSION LINÉAIRE MULTIPLE
# ----------------------------
# C'est le cœur de notre analyse : prédire S4 à partir de S1, S2, S3

print("\n=== 4. RÉGRESSION LINÉAIRE MULTIPLE ===")

# PRÉPARATION DES DONNÉES
# On supprime les lignes qui ont des valeurs manquantes car elles posent problème
donnees_nettoyees_df = donnees_etudiants_df.dropna()
print(f"Données après suppression des NaN: {donnees_nettoyees_df.shape[0]} lignes")

# Séparation des variables explicatives (X) et de la variable à expliquer (y)
# X = ce qu'on utilise pour prédire (S1, S2, S3)
# y = ce qu'on veut prédire (S4)
variables_explicatives_df = donnees_nettoyees_df[['moyenne_semestre_1', 'moyenne_semestre_2', 'moyenne_semestre_3']]
variable_cible_semestre_4 = donnees_nettoyees_df['moyenne_semestre_4']

print(f"Variables explicatives: {variables_explicatives_df.columns.tolist()}")
print(f"Variable à expliquer: moyenne_semestre_4")

# Conversion en tableaux numpy pour les calculs mathématiques
# Pandas est pratique pour manipuler les données, numpy pour les calculs
variables_explicatives_array = variables_explicatives_df.values  # .values convertit un DataFrame en array numpy
variable_cible_array = variable_cible_semestre_4.values

# RÉGRESSION AVEC DONNÉES NORMALISÉES
# La normalisation améliore la stabilité des calculs et permet de comparer les coefficients
print("\n--- Analyse avec données normalisées ---")

# Normalisation des variables explicatives
variables_explicatives_normalisees, moyennes_variables_explicatives, ecarts_types_variables_explicatives = normaliser_donnees(variables_explicatives_array)

# Normalisation de la variable à expliquer
# .reshape(-1, 1) transforme un vecteur en matrice colonne
variable_cible_normalisee, moyenne_variable_cible, ecart_type_variable_cible = normaliser_donnees(variable_cible_array.reshape(-1, 1))
variable_cible_normalisee = variable_cible_normalisee.ravel()  # On remet sous forme de vecteur simple

# Calcul des coefficients de régression avec les données normalisées
coefficients_theta_normalises = calculer_coefficients_regression_lineaire(variables_explicatives_normalisees, variable_cible_normalisee)

# Prédiction avec les données normalisées
predictions_normalisees = predire_variable_cible(variables_explicatives_normalisees, coefficients_theta_normalises)

# Calcul de la qualité du modèle (R²)
coefficient_determination_normalise = calculer_coefficient_correlation_multiple(variable_cible_normalisee, predictions_normalisees)

# Affichage des résultats avec données normalisées
print(f"Coefficients normalisés:")
print(f"Intercept (constante): {coefficients_theta_normalises[0]:.4f}")
for indice, nom_colonne in enumerate(variables_explicatives_df.columns):
    print(f"{nom_colonne}: {coefficients_theta_normalises[indice+1]:.4f}")
print(f"R² (coefficient de détermination): {coefficient_determination_normalise:.4f}")
print(f"Coefficient de corrélation multiple: {np.sqrt(coefficient_determination_normalise):.4f}")

# RÉGRESSION AVEC DONNÉES BRUTES (NON NORMALISÉES)
# Ces coefficients sont plus faciles à interpréter car ils sont dans l'unité originale
print("\n--- Analyse avec données brutes ---")

# Calcul des coefficients avec les données originales
coefficients_theta_bruts = calculer_coefficients_regression_lineaire(variables_explicatives_array, variable_cible_array)

# Prédictions avec les données originales
predictions_brutes = predire_variable_cible(variables_explicatives_array, coefficients_theta_bruts)

# Qualité du modèle (doit être identique aux données normalisées)
coefficient_determination_brut = calculer_coefficient_correlation_multiple(variable_cible_array, predictions_brutes)

# Affichage des résultats avec données brutes
print(f"Coefficients sur données brutes:")
print(f"Intercept (constante): {coefficients_theta_bruts[0]:.4f}")
for indice, nom_colonne in enumerate(variables_explicatives_df.columns):
    print(f"{nom_colonne}: {coefficients_theta_bruts[indice+1]:.4f}")
print(f"R² sur données brutes: {coefficient_determination_brut:.4f}")

# INTERPRÉTATION DES COEFFICIENTS :
# L'équation de prédiction est : S4 = a0 + a1*S1 + a2*S2 + a3*S3
# - a0 (intercept) : valeur de S4 quand S1=S2=S3=0
# - a1 : quand S1 augmente de 1, S4 augmente de a1 (toutes choses égales par ailleurs)
# - a2 : quand S2 augmente de 1, S4 augmente de a2
# - a3 : quand S3 augmente de 1, S4 augmente de a3

# ----------------------------
# 5. ANALYSE PAR DÉPARTEMENT
# ----------------------------
# On analyse les différences entre départements pour voir s'il y a des patterns

print("\n=== 5. ANALYSE PAR DÉPARTEMENT ===")

# Comptage du nombre d'étudiants par département
# .value_counts() compte les occurrences de chaque valeur unique
comptage_etudiants_par_departement = donnees_etudiants_df['dept_etablissement'].value_counts()
print("Nombre d'étudiants par département:")
print(comptage_etudiants_par_departement.head(10))  # On affiche les 10 départements les plus représentés

# Calcul des statistiques de S4 par département
# .groupby() regroupe les données par département
# .agg() applique plusieurs fonctions à la fois (moyenne, compte, écart-type)
statistiques_semestre_4_par_departement_df = donnees_etudiants_df.groupby('dept_etablissement')['moyenne_semestre_4'].agg(
    ['mean', 'count', 'std']).round(3)

# On trie par moyenne décroissante pour voir quels départements ont les meilleures moyennes
statistiques_semestre_4_par_departement_df = statistiques_semestre_4_par_departement_df.sort_values('mean', ascending=False)

# On ne garde que les départements avec au moins 5 étudiants pour avoir des stats fiables
print(f"\nMoyennes au semestre 4 par département (départements avec plus de 5 étudiants):")
statistiques_departements_filtrees_df = statistiques_semestre_4_par_departement_df[statistiques_semestre_4_par_departement_df['count'] >= 5]
print(statistiques_departements_filtrees_df)

# ----------------------------
# 6. VISUALISATIONS GRAPHIQUES
# ----------------------------
# Création de plusieurs graphiques pour mieux comprendre les données

print("\n=== 6. VISUALISATIONS ===")

# Création d'une figure avec 4 sous-graphiques (2x2)
plt.figure(figsize=(12, 8))

# GRAPHIQUE 1 : Boxplots des moyennes par semestre
# Un boxplot montre la distribution d'une variable (médiane, quartiles, valeurs aberrantes)
plt.subplot(2, 2, 1)  # Position : ligne 1, colonne 1
donnees_etudiants_df[colonnes_moyennes_semestrielles].boxplot()
plt.title('Distribution des moyennes par semestre')
plt.xticks(rotation=45)  # Rotation des étiquettes pour éviter le chevauchement

# GRAPHIQUE 2 : Évolution des moyennes globales
# Montre comment évolue la moyenne générale de semestre en semestre
plt.subplot(2, 2, 2)  # Position : ligne 1, colonne 2
# Calcul de la moyenne de chaque semestre sur tous les étudiants
evolution_moyennes_generales = donnees_etudiants_df[colonnes_moyennes_semestrielles].mean()
# Tracé de la courbe d'évolution
plt.plot(range(1, 5), evolution_moyennes_generales, 'bo-', linewidth=2, markersize=8)
plt.title('Évolution de la moyenne générale')
plt.xlabel('Semestre')
plt.ylabel('Moyenne')
plt.grid(True, alpha=0.3)  # Grille en arrière-plan avec transparence

# GRAPHIQUE 3 : Nuage de points S1 vs S4
# Montre la relation entre les moyennes du semestre 1 et du semestre 4
plt.subplot(2, 2, 3)  # Position : ligne 2, colonne 1
plt.scatter(donnees_etudiants_df['moyenne_semestre_1'], donnees_etudiants_df['moyenne_semestre_4'], alpha=0.6)
plt.xlabel('Moyenne Semestre 1')
plt.ylabel('Moyenne Semestre 4')
plt.title('Corrélation S1 vs S4')
plt.grid(True, alpha=0.3)

# GRAPHIQUE 4 : Histogramme du département le plus représenté
# Montre la distribution des notes de S4 pour le département avec le plus d'étudiants
plt.subplot(2, 2, 4)  # Position : ligne 2, colonne 2
departement_plus_represente = comptage_etudiants_par_departement.index[0]  # Département le plus fréquent
# Extraction des données de S4 pour ce département
donnees_departement_principal = donnees_etudiants_df[donnees_etudiants_df['dept_etablissement'] == departement_plus_represente]['moyenne_semestre_4']
# Création de l'histogramme
plt.hist(donnees_departement_principal, bins=15, alpha=0.7, edgecolor='black')
plt.title(f'Distribution S4 - Département {departement_plus_represente}')
plt.xlabel('Moyenne Semestre 4')
plt.ylabel('Fréquence')

# Ajustement automatique de l'espacement entre les graphiques
plt.tight_layout()
plt.show()

# ----------------------------
# 7. COMPARAISON DES DÉPARTEMENTS LES PLUS REPRÉSENTÉS
# ----------------------------
# Analyse détaillée des 3 départements avec le plus d'étudiants

print("\n=== 7. COMPARAISON DES DÉPARTEMENTS LES PLUS REPRÉSENTÉS ===")

# Sélection des 3 départements les plus représentés
trois_departements_principaux = comptage_etudiants_par_departement.head(3).index

# Création d'une figure avec 3 sous-graphiques côte à côte
plt.figure(figsize=(15, 5))

# Pour chaque département, on crée un boxplot des 4 semestres
for indice, departement in enumerate(trois_departements_principaux):
    plt.subplot(1, 3, indice+1)  # 1 ligne, 3 colonnes, position indice+1
    
    # Extraction des données pour ce département
    donnees_departement_courant_df = donnees_etudiants_df[donnees_etudiants_df['dept_etablissement'] == departement]
    
    # Boxplot des 4 semestres pour ce département
    donnees_departement_courant_df[colonnes_moyennes_semestrielles].boxplot()
    plt.title(f'Département {departement} (n={comptage_etudiants_par_departement[departement]})')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ----------------------------
# 8. EXEMPLE DE PRÉDICTION
# ----------------------------
# Démonstration pratique : prédire S4 pour un étudiant fictif

print("\n=== 8. EXEMPLE DE PRÉDICTION ===")

# Création d'un cas d'exemple : étudiant avec S1=12, S2=10, S3=11
# On met ces valeurs dans un tableau numpy (attention à la forme : 1 ligne, 3 colonnes)
exemple_notes_etudiant = np.array([[12, 10, 11]])

# Prédiction avec notre modèle (coefficients sur données brutes)
prediction_moyenne_semestre_4 = predire_variable_cible(exemple_notes_etudiant, coefficients_theta_bruts)
print(f"Prédiction pour un étudiant avec S1=12, S2=10, S3=11:")
print(f"Moyenne prédite au S4: {prediction_moyenne_semestre_4[0]:.2f}")

# CALCUL D'UN INTERVALLE DE CONFIANCE APPROXIMATIF
# Les résidus sont les différences entre les vraies valeurs et les prédictions
residus_predictions = variable_cible_array - predictions_brutes

# L'écart-type des résidus nous donne une idée de la précision de nos prédictions
ecart_type_residus = np.std(residus_predictions)
print(f"Écart-type des résidus: {ecart_type_residus:.2f}")

# Intervalle approximatif : prédiction ± 2 écarts-types
# Cela donne une fourchette dans laquelle la vraie valeur a environ 95% de chances de se trouver
borne_inferieure_intervalle = prediction_moyenne_semestre_4[0] - 2 * ecart_type_residus
borne_superieure_intervalle = prediction_moyenne_semestre_4[0] + 2 * ecart_type_residus
print(f"Intervalle approximatif à 95%: [{borne_inferieure_intervalle:.2f}, {borne_superieure_intervalle:.2f}]")

# ----------------------------
# RÉSUMÉ FINAL
# ----------------------------
# Synthèse de tous les résultats de l'analyse

print("\n=== RÉSUMÉ DE L'ANALYSE ===")
print(f"• {donnees_etudiants_df.shape[0]} étudiants analysés")
print(f"• {len(donnees_etudiants_df['dept_etablissement'].unique())-1} départements représentés")

# Quelle variable est la plus corrélée avec S4 ?
# .iloc[1] car .iloc[0] est S4 lui-même (corrélation = 1)
variable_plus_correlee_avec_s4 = correlations_avec_semestre_4.iloc[1]
nom_variable_plus_correlee = correlations_avec_semestre_4.index[1]
print(f"• Corrélation la plus forte avec S4: {variable_plus_correlee_avec_s4:.3f} ({nom_variable_plus_correlee})")

print(f"• R² du modèle de prédiction: {coefficient_determination_brut:.3f}")
print(f"  → Le modèle explique {coefficient_determination_brut*100:.1f}% de la variabilité de S4")

nombre_etudiants_departement_principal = comptage_etudiants_par_departement[departement_plus_represente]
print(f"• Département le plus représenté: {departement_plus_represente} ({nombre_etudiants_departement_principal} étudiants)")

# INTERPRÉTATION GÉNÉRALE :
# - Plus R² est proche de 1, meilleur est notre modèle
# - Les coefficients nous disent l'importance relative de chaque semestre
# - Les corrélations nous montrent quels semestres sont les plus liés à S4