from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt

datos_1 = np.random.randint(low=0,high=50,size=16).reshape(8,2)
datos_2 = np.random.randint(low=45,high=75,size=16).reshape(8,2)

print("Datos 1:\n", datos_1)
print("Datos 2:\n", datos_2)

datos_total = np.concatenate((datos_1,datos_2))

X = datos_total.transpose()[0]

Y = datos_total.transpose()[1]

plt.figure()
plt.plot(X, Y,'.k')
plt.scatter(X,Y, marker="s", s=50, c=Y, cmap="RdPu")
plt.title("Gráfico")
plt.show()

centros = np.random.randint(low=0,high=75,size=4).reshape(2,2)

# k = cantidad de clústers
# centros = matriz de centros de clúster
# datos = matriz de datos
def k_means(k, centros, datos):
    dist_cluster = []
    labels = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    labels_ant = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    j = 0

    while (labels.all() != labels_ant.all()):
        j += 1
        dist_cluster.clear()
        for i in range(datos.shape[0]):
            dist_cluster.append(np.linalg.norm(datos[i]-centros[0]))
            dist_cluster.append(np.linalg.norm(datos[i]-centros[1]))
        dista = np.array(dist_cluster).reshape(16,2)
        labels_ant = labels.copy()
        labels = np.argmin(dista, axis=1)
        centros = [[0,0],[0,0]]
        cant_c1 = 0
        cant_c2 = 0
        for i in range(labels.shape[0]):
            if labels[i] == 0:
                centros[0] += datos[i]
                cant_c1 += 1
            else:
                centros[1] += datos[i]
                cant_c2 += 1
        centros[0] = centros[0]/cant_c1
        centros[1] = centros[1]/cant_c2
    return dista, labels, j

a,b,c = k_means(2,centros,datos_total)
print(a)
print("\n")
print(b)
print("\n")
print(f'Iteraciones: {c}')