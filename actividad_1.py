import numpy as np
import matplotlib.pyplot as plt


# k = cantidad de clústers
# centros = matriz de centros de clúster
# datos = matriz de datos
def k_means(k, centres, data, dim):
    pre_centres = np.ones(k*dim).reshape(k, dim)
    it = 0

    while (not(np.array_equal(centres, pre_centres))):
        it += 1
        dist_cluster = np.zeros(data.shape[0]*k).reshape(data.shape[0],k)
        mat_pert = np.zeros((data.shape[0]*k), dtype=int).reshape(data.shape[0],k)
        
        for i in range(data.shape[0]):
            for j in range(k):
                dist_cluster[i][j]= np.linalg.norm(data[i]-centres[j])

        print(f"dist_cluster: {dist_cluster}")
        pre_centres = centres.copy()
        centres = np.zeros(k*dim).reshape(k,dim)
        amount_centres = np.zeros((k), dtype=int)
        aux = np.argmin(dist_cluster, axis=1)
        
        print(f"aux: {aux}")

        for i in range(aux.shape[0]):
            mat_pert[i][aux[i]] = 1

        print(f"mat_pert: {mat_pert}")

        for j in range(mat_pert.shape[1]):
            for i in range(mat_pert.shape[0]):
                if mat_pert[i][j] == 1:
                    centres[j] += data[i]
                    amount_centres[j] += 1 
        
        for i in range(amount_centres.shape[0]):
            if amount_centres[i] != 0:
                centres[i] = centres[i]/amount_centres[i]
        
        print(f"--------------------- ITER: {it}")
        print(f"cant_centros: {amount_centres}")
        print(f"centros: {centres}")
        print(f"centros_ant: {pre_centres}")
    
    return dist_cluster, centres, it

#Función para generar datos
def data_generate():
    
    data_1 = np.random.randint(low=0,high=35,size=32).reshape(16,2)
    data_2 = np.random.randint(low=25,high=50,size=32).reshape(16,2)
    
    return np.concatenate((data_1,data_2))


total_data = data_generate()
print(f"Datos totales:\n {total_data}")

init_centres = np.zeros(4).reshape(2, 2)
distancias, centros, iteraciones = k_means(2, init_centres, total_data, 2)

print("\nDistancias al centro de cluster:\n", distancias)
print("\nCentros de cluster:\n", centros)
print(f'\nIteraciones: {iteraciones}')

#----------------------------- GRAFICO -----------------------------
"""
X1 = []; X2 = []; Y1 = []; Y2 = []
for i in range(labels.shape[0]):
    if labels[i] == 0:
        X1.append(datos_total[i][0])
        Y1.append(datos_total[i][1]) 
    else:
        X2.append(datos_total[i][0])
        Y2.append(datos_total[i][1])

W = centros.transpose()[0]
Z = centros.transpose()[1]

plt.figure()
plt.plot(X1, Y1, color='green', marker='o', linestyle='')
plt.plot(X2, Y2, color='blue', marker='o', linestyle='')
plt.plot(W, Z, 'r+')
plt.title("Gráfico")
plt.show()
"""