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


def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients (theta) d'un modèle de régression linéaire multiple.
    - X est une matrice (tableau) contenant les variables explicatives (S1, S2, S3)
    - y est un vecteur contenant la variable à prédire (S4)
    La fonction ajoute une colonne de 1 à X (pour l'interception) et applique la formule des moindres carrés.
    """
    n_samples = X.shape[0]  # nombre d'échantillons (lignes)
    # ajoute une colonne de 1 à gauche de X
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    # calcul des coefficients
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta.flatten()  # retourne un tableau à une dimension


def predire_y(X, theta):
    """
    Prédit les valeurs de y à partir de X et des coefficients theta.
    On ajoute une colonne de 1 à X et on fait le produit matriciel avec theta.
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    y_pred = X_aug @ theta  # prédiction
    return y_pred


def coefficient_correlation_multiple(y_true, y_pred):
    """
    Calcule le coefficient de détermination R² (qualité du modèle).
    R² mesure la proportion de la variance de y expliquée par les variables X.
    """
    y_true = np.ravel(y_true)  # aplatissement des tableaux
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)  # somme des carrés des résidus
    ss_tot = np.sum((y_true - np.mean(y_true))**2)  # somme totale des carrés
    r_squared = 1 - ss_res / ss_tot  # formule de R²
    return r_squared


def normaliser_donnees(X):
    """
    Normalise les données : centre (moyenne 0) et réduit (écart-type 1)
    Cela améliore la stabilité des calculs et la comparabilité entre variables.
    """
    mean = np.mean(X, axis=0)  # moyenne de chaque colonne
    std = np.std(X, axis=0)  # écart-type de chaque colonne
    X_normalized = (X - mean) / std  # transformation
    return X_normalized, mean, std

# ----------------------------
# 1. IMPORT ET EXPLORATION DES DONNÉES
# ----------------------------


print("=== 1. IMPORT ET EXPLORATION DES DONNÉES ===")
df = pd.read_csv('Vue.csv')  # Chargement du fichier CSV dans un DataFrame
# Affiche le nombre de lignes et colonnes
print("Shape du DataFrame:", df.shape)
print("\nPremières lignes:")
print(df.head())  # Affiche les 5 premières lignes du tableau

print("\nInformations sur le dataset:")
print(df.info())  # Affiche les types de colonnes et les valeurs manquantes

print("\nStatistiques descriptives:")
print(df.describe())  # Statistiques : moyenne, min, max, etc.

# Compte des valeurs manquantes dans la colonne 'dept_etablissement'
print(
    f"\nValeurs manquantes dans dept_etablissement: {df['dept_etablissement'].isna().sum()}")

# ----------------------------
# 2. ANALYSE DES CORRÉLATIONS
# ----------------------------

print("\n=== 2. ANALYSE DES CORRÉLATIONS ===")

# On sélectionne les colonnes contenant les moyennes de chaque semestre
numeric_cols = ['moyenne_semestre_1', 'moyenne_semestre_2',
                'moyenne_semestre_3', 'moyenne_semestre_4']
# Calcul de la matrice de corrélation entre ces colonnes
corr_matrix = df[numeric_cols].corr()

print("Matrice de corrélation:")
print(corr_matrix)

# Affichage visuel de la matrice de corrélation
plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Affichage des valeurs numériques dans la matrice
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                 ha='center', va='center', fontsize=10)

# Configuration des axes
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar(im, label='Coefficient de corrélation')
plt.title('Matrice de corrélation entre les moyennes semestrielles')
plt.tight_layout()
plt.show()

# ----------------------------
# 3. CORRÉLATIONS AVEC LE SEMESTRE 4
# ----------------------------

print("\n=== 3. CORRÉLATIONS AVEC LA MOYENNE DU SEMESTRE 4 ===")
correlations_s4 = df[numeric_cols].corr(
)['moyenne_semestre_4'].sort_values(ascending=False)
print("Corrélations avec la moyenne du semestre 4:")
for var, corr in correlations_s4.items():
    if var != 'moyenne_semestre_4':
        print(f"{var}: {corr:.3f}")

# ----------------------------
# 4. RÉGRESSION LINÉAIRE MULTIPLE
# ----------------------------

print("\n=== 4. RÉGRESSION LINÉAIRE MULTIPLE ===")

# On enlève les lignes contenant des valeurs manquantes
df_clean = df.dropna()
print(f"Données après suppression des NaN: {df_clean.shape[0]} lignes")

# Sélection des colonnes pour les variables explicatives et à expliquer
X = df_clean[['moyenne_semestre_1',
              'moyenne_semestre_2', 'moyenne_semestre_3']]
y = df_clean['moyenne_semestre_4']

print(f"Variables explicatives: {X.columns.tolist()}")
print(f"Variable à expliquer: moyenne_semestre_4")

# Conversion en tableau numpy
X_array = X.values
y_array = y.values

# Normalisation des données
X_normalized, X_mean, X_std = normaliser_donnees(X_array)
y_normalized, y_mean, y_std = normaliser_donnees(y_array.reshape(-1, 1))
y_normalized = y_normalized.ravel()

# Calcul des coefficients avec les données normalisées
theta_normalized = coefficients_regression_lineaire(X_normalized, y_normalized)
y_pred_normalized = predire_y(X_normalized, theta_normalized)
r2_normalized = coefficient_correlation_multiple(
    y_normalized, y_pred_normalized)

# Affichage des coefficients normalisés
print(f"\nCoefficients normalisés:")
print(f"Intercept: {theta_normalized[0]:.4f}")
for i, col in enumerate(X.columns):
    print(f"{col}: {theta_normalized[i+1]:.4f}")
print(f"R² (coefficient de détermination): {r2_normalized:.4f}")
print(f"Coefficient de corrélation multiple: {np.sqrt(r2_normalized):.4f}")

# Calcul des coefficients avec les données brutes (non normalisées)
theta_raw = coefficients_regression_lineaire(X_array, y_array)
y_pred_raw = predire_y(X_array, theta_raw)
r2_raw = coefficient_correlation_multiple(y_array, y_pred_raw)

print(f"\nCoefficients sur données brutes:")
print(f"Intercept: {theta_raw[0]:.4f}")
for i, col in enumerate(X.columns):
    print(f"{col}: {theta_raw[i+1]:.4f}")
print(f"R² sur données brutes: {r2_raw:.4f}")

# ----------------------------
# 5. ANALYSE PAR DÉPARTEMENT
# ----------------------------

print("\n=== 5. ANALYSE PAR DÉPARTEMENT ===")

# Compte des étudiants par département
dept_counts = df['dept_etablissement'].value_counts()
print("Nombre d'étudiants par département:")
print(dept_counts.head(10))

# Calcul de la moyenne, nombre et écart-type de S4 par département
dept_means = df.groupby('dept_etablissement')['moyenne_semestre_4'].agg(
    ['mean', 'count', 'std']).round(3)
dept_means = dept_means.sort_values('mean', ascending=False)
print(f"\nMoyennes au semestre 4 par département (départements avec plus de 5 étudiants):")
dept_means_filtered = dept_means[dept_means['count'] >= 5]
print(dept_means_filtered)

# ----------------------------
# 6. VISUALISATIONS GRAPHIQUES
# ----------------------------

print("\n=== 6. VISUALISATIONS ===")

plt.figure(figsize=(12, 8))

# Boxplot : répartition des moyennes par semestre
plt.subplot(2, 2, 1)
df[numeric_cols].boxplot()
plt.title('Distribution des moyennes par semestre')
plt.xticks(rotation=45)

# Évolution des moyennes globales
plt.subplot(2, 2, 2)
mean_evolution = df[numeric_cols].mean()
plt.plot(range(1, 5), mean_evolution, 'bo-', linewidth=2, markersize=8)
plt.title('Évolution de la moyenne générale')
plt.xlabel('Semestre')
plt.ylabel('Moyenne')
plt.grid(True, alpha=0.3)

# Nuage de points entre S1 et S4
plt.subplot(2, 2, 3)
plt.scatter(df['moyenne_semestre_1'], df['moyenne_semestre_4'], alpha=0.6)
plt.xlabel('Moyenne Semestre 1')
plt.ylabel('Moyenne Semestre 4')
plt.title('Corrélation S1 vs S4')
plt.grid(True, alpha=0.3)

# Histogramme du département le plus fréquent
plt.subplot(2, 2, 4)
most_common_dept = dept_counts.index[0]
dept_data = df[df['dept_etablissement'] ==
               most_common_dept]['moyenne_semestre_4']
plt.hist(dept_data, bins=15, alpha=0.7, edgecolor='black')
plt.title(f'Distribution S4 - Département {most_common_dept}')
plt.xlabel('Moyenne Semestre 4')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

# ----------------------------
# 7. COMPARAISON DES DÉPARTEMENTS LES PLUS REPRÉSENTÉS
# ----------------------------

print("\n=== 7. COMPARAISON DES DÉPARTEMENTS LES PLUS REPRÉSENTÉS ===")

top_depts = dept_counts.head(3).index
plt.figure(figsize=(15, 5))

for i, dept in enumerate(top_depts):
    plt.subplot(1, 3, i+1)
    dept_data = df[df['dept_etablissement'] == dept]

    # Boxplot des 4 semestres pour chaque département
    dept_data[numeric_cols].boxplot()
    plt.title(f'Département {dept} (n={dept_counts[dept]})')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ----------------------------
# 8. EXEMPLE DE PRÉDICTION
# ----------------------------

print("\n=== 8. EXEMPLE DE PRÉDICTION ===")

# On crée un étudiant fictif avec des moyennes connues en S1, S2, S3
exemple_etudiant = np.array([[12, 10, 11]])
prediction = predire_y(exemple_etudiant, theta_raw)
print(f"Prédiction pour un étudiant avec S1=12, S2=10, S3=11:")
print(f"Moyenne prédite au S4: {prediction[0]:.2f}")

# On calcule un intervalle de confiance approximatif autour de cette prédiction
residuals = y_array - y_pred_raw
std_residuals = np.std(residuals)
print(f"Écart-type des résidus: {std_residuals:.2f}")
print(
    f"Intervalle approximatif: [{prediction[0]-2*std_residuals:.2f}, {prediction[0]+2*std_residuals:.2f}]")

# ----------------------------
# RÉSUMÉ FINAL
# ----------------------------

print("\n=== RÉSUMÉ DE L'ANALYSE ===")
print(f"• {df.shape[0]} étudiants analysés")
print(f"• {len(df['dept_etablissement'].unique())-1} départements représentés")
print(
    f"• Corrélation la plus forte avec S4: {correlations_s4.iloc[1]:.3f} ({correlations_s4.index[1]})")
print(f"• R² du modèle de prédiction: {r2_raw:.3f}")
print(
    f"• Département le plus représenté: {most_common_dept} ({dept_counts[most_common_dept]} étudiants)")
