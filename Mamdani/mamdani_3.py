#----------------------------------------------------------------------------------------------------------
# Sistema Inteligente de control de tránsito para un cruce peatonal en una autopista.

# • Variables de entrada:

# x1 = Flujo del tránsito sobre la autopista (cantidad de autos por minuto) => BAJO y ALTO.
# x2 = Densidad de peatones que esperan en el cruce (peatones por m²) => BAJA y ALTA.

# • Variables de salida:

# y = Tiempo en que los semáforos se mantienen en luz verde => POCO, MEDIO y MUCHO.


# Rule #1 :

# "IF el flujo del tránsito sobre la autopista es BAJO OR la densidad de peatones que esperan en el cruce es ALTA
# THEN el tiempo en que los semáforos se mantienen en luz verde es POCO."

# Rule #2 :

# "IF el flujo del tránsito sobre la autopista es ALTO AND la densidad de peatones que esperan en el cruce es BAJA
# THEN el tiempo en que los semáforos se mantienen en luz verde es MUCHO."

# Rule #3 :

# "IF el flujo del tránsito sobre la autopista es ALTO AND la densidad de peatones que esperan en el cruce es ALTA
# THEN el tiempo en que los semáforos se mantienen en luz verde es MEDIO."

# Rule #4 :

# "IF el flujo del tránsito sobre la autopista es BAJO AND la densidad de peatones que esperan en el cruce es BAJO
# THEN el tiempo en que los semáforos se mantienen en luz verde es MEDIO."

#----------------------------------------------------------------------------------------------------------
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

people = input('Ingrese cantidad de personas por metro cuadrado:')
cars = input('Ingrese cantidad de autos por minuto:')

# transit_flow = flujo de tránsito
transit_flow = np.arange(0,60)
# flujo de tránsito BAJO
mf_tf_low = fuzz.zmf(transit_flow, 20, 40)
# flujo de tránsito ALTO
mf_tf_high = fuzz.smf(transit_flow, 20, 40)


# people_den = densidad de peatones
people_den = np.arange(0,7)
# densidad de peatones BAJA
mf_pd_low = fuzz.zmf(people_den, 2, 4)
# densidad de peatones ALTA
mf_pd_high = fuzz.smf(people_den, 2, 4)


# time = tiempo en que se mantienen los semáforos en verde en segundos.
time = np.arange(0,60)
# tiempo POCO
mfy_bit = fuzz.zmf(time, 10, 25)
# tiempo MEDIO
mfy_mid = fuzz.gaussmf(time, 30, 10)
# tiempo MUCHO
mfy_lot = fuzz.smf(time, 35, 50)



# Fuzzificación

cars = int(cars)
people = int(people)

rule_1_cars = fuzz.interp_membership(transit_flow, mf_tf_low, cars)
rule_1_people = fuzz.interp_membership(people_den, mf_pd_high, people)

# OR
if (rule_1_cars > rule_1_people): 
    rule_1 = rule_1_cars
else:
    rule_1 = rule_1_people

rule_2_cars = fuzz.interp_membership(transit_flow, mf_tf_high, cars)
rule_2_people = fuzz.interp_membership(people_den, mf_pd_low, people)

# AND
if (rule_2_cars < rule_2_people):
    rule_2 = rule_2_cars
else:
    rule_2 = rule_2_people

rule_3_cars = fuzz.interp_membership(transit_flow, mf_tf_high, cars)
rule_3_people = fuzz.interp_membership(people_den, mf_pd_high, people)

# AND
if (rule_3_cars < rule_3_people):
    rule_3 = rule_3_cars
else:
    rule_3 = rule_3_people

rule_4_cars = fuzz.interp_membership(transit_flow, mf_tf_low, cars)
rule_4_people = fuzz.interp_membership(people_den, mf_pd_low, people)

# AND
if (rule_4_cars > rule_3_people):
    rule_4 = rule_4_cars
else:
    rule_4 = rule_4_people


# Inferencia

output_1 = np.fmin(rule_1, mfy_bit)
output_2 = np.fmin(rule_2, mfy_lot)
output_3 = np.fmin(rule_3, mfy_mid)
output_4 = np.fmin(rule_4, mfy_mid)


# Agregación

aggregated = np.fmax(output_4, np.fmax(output_3, np.fmax(output_1, output_2)))


# Desfuzzificación

result = fuzz.defuzz(time, aggregated, 'centroid')
print(f'Tiempo en verde: {round(result,2)} segundos.')


# Gráfico

y_result = fuzz.interp_membership(time, aggregated, result)
yprima = np.zeros_like(time)
plt.plot(time, mfy_bit, linestyle='--', color='red')
plt.plot(time, mfy_mid, linestyle='--', color='green')
plt.plot(time, mfy_lot, linestyle='--', color='blue')
plt.fill_between(time, yprima, aggregated, facecolor='pink', linestyle='--', alpha=0.7)
plt.plot([result, result], [0, y_result], 'k', linewidth=1.5, alpha=0.9)
plt.plot(time, aggregated, linestyle='--', color='hotpink')
plt.show()