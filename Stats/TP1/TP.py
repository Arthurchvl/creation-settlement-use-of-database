import pandas as pd
import numpy as np
import math as mt
#%%
chemin_fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Stats/TP1/Sangliers.csv" 

df = pd.read_csv(chemin_fichier)
#print(df.head())

nb_sanglier_preleves = df["Nb_sanglier_preleves"].to_numpy()
nb_permis_chasse = df["Nb_permis_chasse"].to_numpy()
#%%
def moyenne(x):
    return np.sum(x)/len(x)

#print(moyenne(nb_sanglier_preleves))
#print(np.mean(nb_sanglier_preleves))
#%%
def variance(x):
    return moyenne(x**2) - (moyenne(x)**2)

#print(variance(nb_sanglier_preleves))
#print(np.var(nb_sanglier_preleves))
#%%
def ecart_type(x):
    return mt.sqrt(np.var(x))
#print(ecart_type(nb_sanglier_preleves))
#print(np.std(nb_sanglier_preleves, ddof=0))
#%%
def covariance(x, y):
    return np.mean(x*y) - (np.mean(x)*np.mean(y))
#print(covariance(nb_permis_chasse, nb_sanglier_preleves))
#print(np.cov(nb_permis_chasse, nb_sanglier_preleves, ddof=0)[0,1])
#%%
def correlation(x, y):
    return np.cov(x,y, ddof=0)[0,1] / (np.std(x, ddof=0)*np.std(y, ddof=0))
#print(correlation(nb_permis_chasse, nb_sanglier_preleves))
print(np.corrcoef(nb_permis_chasse, nb_sanglier_preleves)[0,1])
#%%
