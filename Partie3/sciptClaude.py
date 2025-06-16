import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonctions personnalisées pour la régression linéaire
def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients de l'hyperplan pour une régression linéaire multiple.
    X : ndarray de shape (n, m)
    y : ndarray de shape (n, 1) ou (n,)
    Retourne : theta (ndarray de shape (m+1,) avec b à l'indice 0)
    """
    n_samples = X.shape[0]
    # Ajouter une colonne de 1 pour l'ordonnée à l'origine
    X_aug = np.hstack((np.ones((n_samples, 1)), X))  # shape: (n, m+1)
    # Formule des moindres carrés
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta.flatten()

def predire_y(X, theta):
    """
    Calcule y_pred à partir de X et theta.
    X : ndarray de shape (n, m)
    theta : ndarray de shape (m+1,) — inclut l'intercept
    Retourne : y_pred (ndarray de shape (n,))
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))  # ajoute une colonne de 1
    y_pred = X_aug @ theta
    return y_pred

def coefficient_correlation_multiple(y_true, y_pred):
    """
    Calcule le coefficient de corrélation multiple (R^2)
    y_true : valeurs réelles (shape: (n,))
    y_pred : valeurs prédites (shape: (n,))
    Retourne : R² (float)
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared

def normaliser_donnees(X):
    """
    Normalise les données (centrage et réduction)
    X : ndarray
    Retourne : X_normalized, mean, std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

# 1. Import des données
print("=== 1. IMPORT ET EXPLORATION DES DONNÉES ===")
df = pd.read_csv('/Users/arthurchauvel/Desktop/cours/SAES/S204/Partie3/Vue.csv')
print("Shape du DataFrame:", df.shape)
print("\nPremières lignes:")
print(df.head())

print("\nInformations sur le dataset:")
print(df.info())

print("\nStatistiques descriptives:")
print(df.describe())

# Gestion des valeurs manquantes dans dept_etablissement
print(f"\nValeurs manquantes dans dept_etablissement: {df['dept_etablissement'].isna().sum()}")

# 2. Analyse des corrélations
print("\n=== 2. ANALYSE DES CORRÉLATIONS ===")

# Créer une matrice de corrélation pour les variables numériques
numeric_cols = ['moyenne_semestre_1', 'moyenne_semestre_2', 'moyenne_semestre_3', 'moyenne_semestre_4']
corr_matrix = df[numeric_cols].corr()

print("Matrice de corrélation:")
print(corr_matrix)

# Visualisation de la matrice de corrélation avec matplotlib
plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Ajouter les valeurs dans les cellules
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

# 3. Analyse des corrélations avec le semestre 4 (objectif de réussite)
print("\n=== 3. CORRÉLATIONS AVEC LA MOYENNE DU SEMESTRE 4 ===")
correlations_s4 = df[numeric_cols].corr()['moyenne_semestre_4'].sort_values(ascending=False)
print("Corrélations avec la moyenne du semestre 4:")
for var, corr in correlations_s4.items():
    if var != 'moyenne_semestre_4':
        print(f"{var}: {corr:.3f}")

# 4. Régression linéaire multiple
print("\n=== 4. RÉGRESSION LINÉAIRE MULTIPLE ===")

# Préparation des données (suppression des NaN)
df_clean = df.dropna()
print(f"Données après suppression des NaN: {df_clean.shape[0]} lignes")

# Variables explicatives (S1, S2, S3) et variable à expliquer (S4)
X = df_clean[['moyenne_semestre_1', 'moyenne_semestre_2', 'moyenne_semestre_3']]
y = df_clean['moyenne_semestre_4']

print(f"Variables explicatives: {X.columns.tolist()}")
print(f"Variable à expliquer: moyenne_semestre_4")

# Conversion en arrays numpy
X_array = X.values
y_array = y.values

# Normalisation des données
X_normalized, X_mean, X_std = normaliser_donnees(X_array)
y_normalized, y_mean, y_std = normaliser_donnees(y_array.reshape(-1, 1))
y_normalized = y_normalized.ravel()

# Régression linéaire sur données normalisées
theta_normalized = coefficients_regression_lineaire(X_normalized, y_normalized)
y_pred_normalized = predire_y(X_normalized, theta_normalized)
r2_normalized = coefficient_correlation_multiple(y_normalized, y_pred_normalized)

print(f"\nCoefficients normalisés:")
print(f"Intercept: {theta_normalized[0]:.4f}")
for i, col in enumerate(X.columns):
    print(f"{col}: {theta_normalized[i+1]:.4f}")
print(f"R² (coefficient de détermination): {r2_normalized:.4f}")
print(f"Coefficient de corrélation multiple: {np.sqrt(r2_normalized):.4f}")

# Régression sur données non normalisées pour l'interprétation
theta_raw = coefficients_regression_lineaire(X_array, y_array)
y_pred_raw = predire_y(X_array, theta_raw)
r2_raw = coefficient_correlation_multiple(y_array, y_pred_raw)

print(f"\nCoefficients sur données brutes:")
print(f"Intercept: {theta_raw[0]:.4f}")
for i, col in enumerate(X.columns):
    print(f"{col}: {theta_raw[i+1]:.4f}")
print(f"R² sur données brutes: {r2_raw:.4f}")

# 5. Analyse par département
print("\n=== 5. ANALYSE PAR DÉPARTEMENT ===")

# Compter les étudiants par département
dept_counts = df['dept_etablissement'].value_counts()
print("Nombre d'étudiants par département:")
print(dept_counts.head(10))

# Moyennes par département pour le semestre 4
dept_means = df.groupby('dept_etablissement')['moyenne_semestre_4'].agg(['mean', 'count', 'std']).round(3)
dept_means = dept_means.sort_values('mean', ascending=False)
print(f"\nMoyennes au semestre 4 par département (départements avec plus de 5 étudiants):")
dept_means_filtered = dept_means[dept_means['count'] >= 5]
print(dept_means_filtered)

# 6. Visualisations
print("\n=== 6. VISUALISATIONS ===")

# Evolution des moyennes par semestre
plt.figure(figsize=(12, 8))

# Boxplot des moyennes par semestre
plt.subplot(2, 2, 1)
df[numeric_cols].boxplot()
plt.title('Distribution des moyennes par semestre')
plt.xticks(rotation=45)

# Evolution moyenne générale
plt.subplot(2, 2, 2)
mean_evolution = df[numeric_cols].mean()
plt.plot(range(1, 5), mean_evolution, 'bo-', linewidth=2, markersize=8)
plt.title('Évolution de la moyenne générale')
plt.xlabel('Semestre')
plt.ylabel('Moyenne')
plt.grid(True, alpha=0.3)

# Scatter plot S1 vs S4
plt.subplot(2, 2, 3)
plt.scatter(df['moyenne_semestre_1'], df['moyenne_semestre_4'], alpha=0.6)
plt.xlabel('Moyenne Semestre 1')
plt.ylabel('Moyenne Semestre 4')
plt.title('Corrélation S1 vs S4')
plt.grid(True, alpha=0.3)

# Histogramme département le plus représenté
plt.subplot(2, 2, 4)
most_common_dept = dept_counts.index[0]
dept_data = df[df['dept_etablissement'] == most_common_dept]['moyenne_semestre_4']
plt.hist(dept_data, bins=15, alpha=0.7, edgecolor='black')
plt.title(f'Distribution S4 - Département {most_common_dept}')
plt.xlabel('Moyenne Semestre 4')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

# 7. Comparaison des performances par département (top 3)
print("\n=== 7. COMPARAISON DES DÉPARTEMENTS LES PLUS REPRÉSENTÉS ===")

top_depts = dept_counts.head(3).index
plt.figure(figsize=(15, 5))

for i, dept in enumerate(top_depts):
    plt.subplot(1, 3, i+1)
    dept_data = df[df['dept_etablissement'] == dept]
    
    # Boxplot des 4 semestres pour ce département
    dept_data[numeric_cols].boxplot()
    plt.title(f'Département {dept} (n={dept_counts[dept]})')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 8. Prédiction d'exemple
print("\n=== 8. EXEMPLE DE PRÉDICTION ===")
# Exemple: étudiant avec moyennes S1=12, S2=10, S3=11
exemple_etudiant = np.array([[12, 10, 11]])
prediction = predire_y(exemple_etudiant, theta_raw)
print(f"Prédiction pour un étudiant avec S1=12, S2=10, S3=11:")
print(f"Moyenne prédite au S4: {prediction[0]:.2f}")

# Intervalle de confiance approximatif
residuals = y_array - y_pred_raw
std_residuals = np.std(residuals)
print(f"Écart-type des résidus: {std_residuals:.2f}")
print(f"Intervalle approximatif: [{prediction[0]-2*std_residuals:.2f}, {prediction[0]+2*std_residuals:.2f}]")

print("\n=== RÉSUMÉ DE L'ANALYSE ===")
print(f"• {df.shape[0]} étudiants analysés")
print(f"• {len(df['dept_etablissement'].unique())-1} départements représentés") # -1 pour les NaN
print(f"• Corrélation la plus forte avec S4: {correlations_s4.iloc[1]:.3f} ({correlations_s4.index[1]})")
print(f"• R² du modèle de prédiction: {r2_raw:.3f}")
print(f"• Département le plus représenté: {most_common_dept} ({dept_counts[most_common_dept]} étudiants)")