#Produces a sample pose_prior file (REQUIRED FOR AVATAR MODEL)
import numpy as np

num_joints = 24
dim = num_joints * 3  # 3 dimensiones por joint
num_components = 1

# Media: vector de ceros
mean = np.zeros(dim)

# Covarianza: matriz identidad (puedes escalarla si quieres)
covariance = np.eye(dim)

with open("../data/avatar-model/pose_prior.txt", "w") as f:
    # Número de componentes y dimensión
    f.write(f"{num_components} {dim}\n")
    
    # Pesos de cada componente (1.0 para uno solo)
    for _ in range(num_components):
        f.write("1.0\n")
        
    # Medias de cada componente
    for _ in range(num_components):
        f.write(" ".join(map(str, mean)) + "\n")
        
    # Covarianzas de cada componente (dim líneas por componente)
    for _ in range(num_components):
        for i in range(dim):
            f.write(" ".join(map(str, covariance[i])) + "\n")