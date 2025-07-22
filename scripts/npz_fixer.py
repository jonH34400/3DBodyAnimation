#Simple Script to preprocess npz model to work with avatar model
import numpy as np

model = np.load('../data/avatar-model/model.npz', allow_pickle=True)
kintree = model['kintree_table']

# Reescribe para que root tenga padre -1
new_kintree = kintree.copy()
parent = kintree[0]
child = kintree[1]

for i in range(len(parent)):
    if parent[i] == child[i]:
        new_kintree[0, i] = -1  # Establecer root

# Sobrescribe archivo
np.savez('../data/avatar-model/model_fixed.npz', **{**model, 'kintree_table': new_kintree})
