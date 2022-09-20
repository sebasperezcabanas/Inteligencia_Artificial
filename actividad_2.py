from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("datos_act_2.txt", sep="  ", names=['X','Y'])
X = df[['X']]
Y = df[['Y']]

mat = df.to_numpy()

#plt.figure()
#plt.title("Actividad 2")
#plt.plot(X, Y, color="darkmagenta", marker='o', linestyle='', label='_nolegend_')
#plt.show()

colores = ["hotpink","tomato","seagreen"]
colores_in = ["lightpink","lightsalmon","lightgreen"]

N = 3
km = KMeans(N)
etiquetapuntos = km.fit_predict(df)
grupos, counts = np.unique(etiquetapuntos, return_counts=True)
centros = km.cluster_centers_


intracluster_distance = np.zeros(mat.shape[0]*N).reshape(mat.shape[0],N)

for gp in grupos:
    for i in range(mat.shape[0]):
        if etiquetapuntos[i] == gp:
            intracluster_distance[i][gp]= np.linalg.norm(mat[i]-centros[gp])

"""
print(intracluster_distance)
maximo = np.argmax(intracluster_distance, axis=0)
print(maximo)
"""
for gp in grupos:
    x = []
    y = []
    for i in range(mat.shape[0]):
        if etiquetapuntos[i] == gp:
            x.append(mat[i][0])
            y.append(mat[i][1])
        plt.plot(x,y,'.', color=colores[gp], label='_nolegend_')


maximo = np.array([])
for gp in grupos:
    twenty_percent = int(counts[gp]/5)
    for i in range(twenty_percent):
        maximo = np.argmax(intracluster_distance, axis=0)
        intracluster_distance[maximo[gp]][gp] = 0
        etiquetapuntos[maximo[gp]] = 10
         

for gp in grupos:
    x = []
    y = []
    for i in range(mat.shape[0]):
        if etiquetapuntos[i] == gp:
            x.append(mat[i][0])
            y.append(mat[i][1])
        plt.plot(x,y,'.', color=colores_in[gp], label='_nolegend_')

plt.plot(centros[:,0],centros[:,1], '.', color="black", markersize=10)
plt.legend(["Centros de clúster"], loc='lower right')
plt.show()