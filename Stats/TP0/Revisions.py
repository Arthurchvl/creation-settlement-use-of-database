import numpy as np
#%%
A = np.array([[1, 2, 3], [4, 5, 6]])
# elle renvoie : array([[1, 2, 3],
#                       [4, 5, 6]])
#%%
B=np.zeros((2,3))
# elle renvoie : array([[0., 0., 0.],
#                       [0., 0., 0.]])
#%%
C=np.ones((3,2))
# array([[1., 1.],
#      [1., 1.],
#      [1., 1.]])
#%%
D=np.eye(4)
# array([[1., 0., 0., 0.],
#        [0., 1., 0., 0.],
#        [0., 0., 1., 0.],
#        [0., 0., 0., 1.]])
#%%
M1 = np.array([5, 6, 7, 8, 9])
#%%
M2 = np.array([1, 1, 1, 1, 1])
#%%
M3 = np.array([[1], [2], [3]])
#%%
M4 = np.array([[1, 2, 4], [-1, 2, 3], [1, 8, 9]])
#%%
L3 = M3.tolist()
print(len(L3), len(M3))
print(np.shape(L3), np.shape(M3))
# Les résultats sont identiques pours L3 et M3
#%%
np.concatenate((A,B),axis=0)
# array([[1., 2., 3.],
#        [4., 5., 6.],
#        [0., 0., 0.],
#        [0., 0., 0.]])
#%%
np.concatenate((A,B),axis=1)
# array([[1., 2., 3., 0., 0., 0.],
#        [4., 5., 6., 0., 0., 0.]])
#%%
np.concatenate((B,C),axis=0)
# ValueError: all the input array dimensions except for the concatenation axis must match exactly,
# but along dimension 1, the array at index 0 has size 3 and the array at index 1 has size 2
#%% 
# M4[1, 0] renvoie np.int64(-1)
# M4[0, :] renvoie array([1, 2, 4])
# M4[0:1, :] renvoie array([[1, 2, 4]])
# M4[ :, 1] renvoie array([2, 2, 8])
# M4[1:3, 0 :2] renvoie array([[-1,  2],
#                              [ 1,  8]])
#%%
# np.shape(M4[0, :]) renvoie (3,)
# np.shape(M4[0:1, :]) renvoie (1, 3)

# np.shape(A[:,1]) renvoie (2,)
# np.shape(A[ :, 1 :2]) renvoie (2, 1)

# Les valeurs renvoyées sont différentes,
# de plus dans le 1er np.shape; le 3 ou le 2 est au début de la liste retournée alors qu'il est à la fin dans la 2eme
#%%
M5 = np.concatenate((M2,M1),axis=0)
#%%
M6 = np.array([M1, M2])
#%%
M7 = M4[1:, :2]
#%%
M8 = M4[[2, 0, 1]]
#%%
M9 = M1
M9 = M9[M9[0] == 8]
#%%
Mtemp = np.array([2])
M10 = np.concatenate((Mtemp, M2), axis=0)
#%%
M11 = M3
M11 = M11 + M11[:1] + 9
#%%
M12 = M4
M12 = np.delete(M12, 1, 0)
#%%
M13 = M4
M13 = np.delete(M13, 0, 1)