import numpy as np
import pandas as pd
#%%
"""
La varibale endogène est la variable la quantité des dégâts causés par les sangliers.
Les variable explicatives sont le nombre de morts causés par la chasse et la consommation de viande de porc.
"""
#%%
fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Stats/TP2/Sangliers.csv"

Sa_DF = pd.read_csv(fichier)
Sa_AR = Sa_DF.to_numpy().astype(np.float64)
print(Sa_DF.head())
print(Sa_AR)
#%%
Sa0_AR = Sa_AR[:, 1:]
print(Sa0_AR)
#%%
Sa0_AR_N = Sa0_AR
for i in range(0, 4):
    Sa0_AR_N = (Sa_AR - np.mean(Sa0_AR)) / np.std(Sa_AR, ddof = 0)
print(Sa0_AR_N)
print("")
print(Sa0_AR)
#%%
Y = Sa0_AR[:, 1]
print(Y)
#%%
X = Sa0_AR[:, [0, 2, 3]]
print(X)
#%%
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


theta = coefficients_regression_lineaire(X, Y)
print(theta)

# %%
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


y_pred = predire_y(X, theta)
print(y_pred)

# %%


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


print(coefficient_correlation_multiple(Y, y_pred))