import pandas as pd
import numpy as np
#%%
fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Stats/TP3/Vue2.csv"
VueDF = pd.read_csv(fichier)
#%%
Vue = VueDF.to_numpy()
Vue2Str = Vue[:, :3]
Vue2Num = Vue[:, 3:]
print(Vue2Num)
#%%
Lignes_ma_categorie = np.argwhere(Vue2Str[:, 1] == "Cadre administratif et commercial d'entr")
print(Lignes_ma_categorie)
#%%
Vue2Num[Lignes_ma_categorie, 0]
#%%
Vue2Num[Lignes_ma_categorie, 1]
