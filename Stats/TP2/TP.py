import numpy as np
import math
from sklearn.linear_model import LinearRegression
import pandas as pd
#%%
"""
La varibale endogène est la variable la quantité des dégâts causés par les sangliers.
Les variable explicatives sont le nombre de morts causés par la chasse et la consommation de viande de porc.
"""
#%%
fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Stats/TP2/Sangliers.csv"

Sa_DF = pd.read_csv(fichier)
Sa_AR = Sa_DF.to_numpy()

print(Sa_AR)