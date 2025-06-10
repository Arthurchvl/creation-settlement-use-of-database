import pandas as pd
import numpy as np

#%%
fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Stats/TP3/Vue.csv"
VueDF = pd.read_csv(fichier)
#%%
Vue = VueDF.to_numpy()
VueStr = Vue[:, :3].astype(str)
VueNum = Vue[:, 3:].astype(float)
print(VueStr.dtype)
#%%
Correls=np.corrcoef(VueNum,rowvar=False)
print(Correls)
#%%
# 1 7 8 11 12 13 17
# soit la m1101, m2101, m2102, m2105, m2106, m2107
# Les matières les plus importantes sont
# - Introduction aux systèmes informatique
# - Mathèmatiques discrètes
# - Algèbre Linéraire
# - Communication
# - PPP
#%%
# m1101,m1102,m1103,m1104
#%%
Y = VueNum[:, 16] 
X = VueNum[:, 1:5]
print(X)
#%%
Yn = Y
for i in range(0, 4):
    Yn = (Y - np.mean(Y)) / np.std(Y, ddof = 0)

Xn = X
for i in range(0, 4):
    Xn = (X - np.mean(X)) / np.std(X, ddof = 0)
    
print(X)
print("")
print(Y)
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