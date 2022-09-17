import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------

# Función que realiza el algoritmo K-means:
# k = cantidad de clústers
# centres = matriz de centros de clúster
# data = matriz de datos
# dim = cantidad de features
def k_means(k, centres, data, dim):
    pre_centres = np.ones(k*dim).reshape(k, dim)
    it = 0

    while (not(np.array_equal(centres, pre_centres))):
        it += 1
        # Inicializa la matriz de distancias de cada punto a cada clúster
        # filas => dato
        # columnas => clúster
        dist_cluster = np.zeros(data.shape[0]*k).reshape(data.shape[0],k)

        # Inicializa matriz de pertenecias
        mat_pert = np.zeros((data.shape[0]*k), dtype=int).reshape(data.shape[0],k)
        
        # Calcula la distancia euclidiana de cada punto a cada clúster
        for i in range(data.shape[0]):
            for j in range(k):
                dist_cluster[i][j]= np.linalg.norm(data[i]-centres[j])

        pre_centres = centres.copy()

        # Resetea matriz de centros de clúster
        centres = np.zeros(k*dim).reshape(k,dim)
        # Inicializa contadores de cantidad de datos pertenecientes a cada clúster
        amount_centres = np.zeros((k), dtype=int)

        # Guarda a qué clúster pertenece cada dato
        aux = np.argmin(dist_cluster, axis=1)
        
        # Arma la matriz de pertenencias
        for i in range(aux.shape[0]):
            mat_pert[i][aux[i]] = 1

        # Acumula los datos pertenecientes a cada clúster y los suma
        for j in range(mat_pert.shape[1]):
            for i in range(mat_pert.shape[0]):
                if mat_pert[i][j] == 1:
                    centres[j] += data[i]
                    amount_centres[j] += 1 
        
        # Divide la acumulación por la cantidad de datos pertenecientes al clúster
        for i in range(amount_centres.shape[0]):
            if amount_centres[i] != 0:
                centres[i] = centres[i]/amount_centres[i]
        
        #print(f"--------------------- ITER: {it}")
        #print(f"cant_centros: {amount_centres}")
        #print(f"centros: {centres}")
        #print(f"centros_ant: {pre_centres}")
    
    return dist_cluster, centres, mat_pert, it

# -------------------------------------------------------------------------------------------------------------

# Función para generar datos:
def data_generate(total_data, min_range, max_range, features, amount_data):
    
    data = np.random.randint(low=min_range,high=max_range,size=features*amount_data).reshape(amount_data,features)

    if np.sum(total_data)==0:
        total_data = data.copy()
    else:
        total_data = np.concatenate((total_data,data))

    return total_data

# -------------------------------------------------------------------------------------------------------------

features = 2 # cantidad de características => D
amount_data = 100 # cantidad de datos => N
amount_batch = 5 # cantidad de lotes de datos
k = 5 # cantidad de clústers

total_data = np.zeros(amount_data*amount_batch*features).reshape(amount_data*amount_batch,features)

# Lote 1:
total_data = data_generate(total_data,0,25,features,amount_data)
# Lote 2:
total_data = data_generate(total_data,25,50,features,amount_data)
# Lote 3:
total_data = data_generate(total_data,50,75,features,amount_data)
# Lote 4:
total_data = data_generate(total_data,75,100,features,amount_data)
# Lote 5:
total_data = data_generate(total_data,100,125,features,amount_data)

print(f"Datos totales:\n {total_data}")


# Inicialización de los centros de clúster en [0,0..n]
init_centres = np.zeros(k*features).reshape(k, features)

distances, centres, mat_pert, it = k_means(k, init_centres, total_data, features)

print(f"\nDistancias a los centros de clúster:\n {distances}")
print(f"\nCentros de clúster:\n {centres}")
print(f"\nMatriz de pertenencias:\n {mat_pert}")
print(f"\nIteraciones: {it}")


# ------------------------------------------ G R A F I C O ----------------------------------------------------

colores = ["green","red","yellow","blue","orange","grey"]

i_cluster = np.argmax(mat_pert, axis=1)

X1 = []; Y1 = []
X2 = []; Y2 = []
X3 = []; Y3 = []
X4 = []; Y4 = []
X5 = []; Y5 = []

for i in range(i_cluster.shape[0]):
    if i_cluster[i] == 0:
        X1.append(total_data[i][0]); Y1.append(total_data[i][1]) 
    elif i_cluster[i] == 2:
        X3.append(total_data[i][0]); Y3.append(total_data[i][1])
    elif i_cluster[i] == 3:
        X4.append(total_data[i][0]); Y4.append(total_data[i][1])
    elif i_cluster[i] == 4:
        X5.append(total_data[i][0]); Y5.append(total_data[i][1])
    else:
        X2.append(total_data[i][0]); Y2.append(total_data[i][1])
    """
    """

plt.figure()
plt.title("Gráfico")
plt.plot(X1, Y1, color="darkmagenta", marker='o', linestyle='', label='_nolegend_')
plt.plot(X2, Y2, color='darkcyan', marker='o', linestyle='', label='_nolegend_')
plt.plot(X3, Y3, color='darkred', marker='o', linestyle='', label='_nolegend_')
plt.plot(X4, Y4, color='peru', marker='o', linestyle='', label='_nolegend_')
plt.plot(X5, Y5, color='slategray', marker='o', linestyle='', label='_nolegend_')

W = centres.transpose()[0]
Z = centres.transpose()[1]
plt.plot(W, Z, '.k', color="red", markersize=15)
plt.legend(["Centros de clúster"], loc='lower right')
plt.show()