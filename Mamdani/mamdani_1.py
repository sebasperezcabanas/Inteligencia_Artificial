import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# Crea las membership function para la entrada y la salida.
# Cada mfx|mfy es un vector donde cada elemento contiene el valor de μ para ese x|y entre [mín,máx]
# - máx no se incluye, es el límite.
 
x = np.arange(-21,21)
mfx_1 = fuzz.trapmf(x, [-20, -15, -6, -3])
mfx_2 = fuzz.trapmf(x, [-6, -3, 3, 6])
mfx_3 = fuzz.trapmf(x, [3, 6, 15, 20])

y = np.arange(-3,16)
mfy_1 = fuzz.trapmf(y, [-2.46, -1.46, 1.46, 2.46])
mfy_2 = fuzz.trapmf(y, [1.46, 2.46, 5, 7])
mfy_3 = fuzz.trapmf(y, [5, 7, 13, 15])


# Fuzzificación: entro con un valor de la variable difusa en las mfx (en el dominio [eje x]) y obtengo
# el valor de verdad [rule] de la proposición para ese x [value].

value = 5
rule_1 = fuzz.interp_membership(x, mfx_1, value)
rule_2 = fuzz.interp_membership(x, mfx_2, value)
rule_3 = fuzz.interp_membership(x, mfx_3, value)


# Inferencia: a partir del grado de verdad en que se activa la regla obtengo los posibles valores de verdad en y.

output_1 = np.fmin(rule_1, mfy_1)
output_2 = np.fmin(rule_2, mfy_2)
output_3 = np.fmin(rule_3, mfy_3)


# Agregación: se combinan los resultados de todas las reglas.

aggregated = np.fmax(output_1, np.fmax(output_2,output_3))


# Desfuzzificación: por el método del centroide.

result = fuzz.defuzz(y, aggregated, 'centroid')
print(f'Resultado: {result}')


# Gráfico

y_result = fuzz.interp_membership(y, aggregated, result)
yprima = np.zeros_like(y)
plt.plot(y, mfy_1, linestyle='--', color='red')
plt.plot(y, mfy_2, linestyle='--', color='green')
plt.plot(y, mfy_3, linestyle='--', color='blue')
plt.fill_between(y, yprima, aggregated, facecolor='pink', linestyle='--', alpha=0.7)
plt.plot([result, result], [0, y_result], 'k', linewidth=1.5, alpha=0.9)
plt.plot(y, aggregated, linestyle='--', color='hotpink')
plt.show()