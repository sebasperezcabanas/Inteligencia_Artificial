from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt


# k = cantidad de clústers
# centros = matriz de centros de clúster
# datos = matriz de datos
def k_means(k, centros, datos):
    dist_cluster = []
    labels = []
    centros_ant = np.array([[1,1],[1,1]]).reshape(2, 2)
    j = 0

    while (not(np.array_equal(centros, centros_ant))):
        j += 1
        dist_cluster.clear()
        for i in range(datos.shape[0]):
            dist_cluster.append(np.linalg.norm(datos[i]-centros[0]))
            dist_cluster.append(np.linalg.norm(datos[i]-centros[1]))
        dista = np.array(dist_cluster).reshape(32,2)
        labels = np.argmin(dista, axis=1)
        
        centros_ant = centros.copy()
        centros = np.array([[0,0],[0,0]])
        cant_c1 = 0
        cant_c2 = 0
        for i in range(labels.shape[0]):
            if labels[i] == 0:
                centros[0] += datos[i]
                cant_c1 += 1
            else:
                centros[1] += datos[i]
                cant_c2 += 1
        
        if cant_c1 != 0:
            centros[0] = centros[0]/cant_c1
        if cant_c2 != 0:
            centros[1] = centros[1]/cant_c2
    
    return dista, labels, centros, j


datos_1 = np.random.randint(low=0,high=35,size=32).reshape(16,2)
datos_2 = np.random.randint(low=25,high=50,size=32).reshape(16,2)
datos_total = np.concatenate((datos_1,datos_2))

print("Datos totales:\n", datos_total)

centrosIni = np.array([[0,0],[0,0]]).reshape(2, 2)
distancias, labels, centros, iteraciones = k_means(2, centrosIni, datos_total)

print("\nDistancias al centro de cluster:\n", distancias)
print("\nCentros de cluster:\n", centros)
print("\nLabels:\n", labels)
print(f'\nIteraciones: {iteraciones}')

#----------------------------- GRAFICO -----------------------------

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