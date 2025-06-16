import pandas as pd
import numpy as np
#%%
fichier = "/Users/arthurchauvel/Desktop/cours/SAES/S204/Partie3/Vue.csv"
VueDF = pd.read_csv(fichier)
#%%
Vue = VueDF.to_numpy().astype(float)
print(Vue.dtype)
#%%
Correls=np.corrcoef(Vue,rowvar=False)