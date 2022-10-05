import numpy as np

# Función de evaluación
def f(x):
    return (300 - (x - 15)**2)

# ----------------------------------------- Población inicial -----------------------------------------

# Crea la población inicial
def create_pop(init):
    pop = []
    
    for i in range(init):
        x = np.random.randint(low=0, high=2, size=5)
        pop.append(x)
    pop = np.array(pop).reshape(6,5)

    return pop


n_init_pop = 6

pop = create_pop(n_init_pop)
values, count = np.unique(pop, axis=0, return_counts=True)

# Controlo que no haya individuos repetidos
while(len(count) != n_init_pop):
    pop = create_pop(n_init_pop)
    values, count = np.unique(pop, axis=0, return_counts=True)

print(f'Población inicial: \n{pop}')

# ---------------------------------------- Selección de padres ----------------------------------------

# Convierte de binario a decimal
def bin_to_dec(number):
	decimal_num = 0 

	for posicion, digito_string in enumerate(number[::-1]):
		decimal_num += int(digito_string) * 2 ** posicion

	return decimal_num

# Calcula el valor de la f(x) para cada individuo
def calculate_fitness(individual):
    cad = ''

    for i in range(len(individual)):
        num = str(individual[i])
        cad += num
    number = bin_to_dec(cad)
    return f(number)

# Sumo todas las fitness
sum = 0
for i in range(pop.shape[0]):
    sum += calculate_fitness(pop[i])
    print(f'\nf(x): {calculate_fitness(pop[i])}')

# Elije el mejor individuo 
def best(population):
    max = 0
    percentage = 0.0
    ind_max = []
    i_max = 0

    for i in range(population.shape[0]):
        fit_indiv = calculate_fitness(population[i])
        percentage = (fit_indiv*100)/sum
        if ( percentage > max):
            max = percentage
            ind_max = population[i]
            i_max = i
    
    return ind_max, i_max


_max_,i_max = best(pop)
print(f'El mejor individuo es: {_max_}')

# Saca el mejor individuo para tomar el segundo mejor
sub_pop = np.delete(pop,i_max,axis=0)
_max2_,i2_max = best(sub_pop)
print(f'El segundo mejor individuo es: {_max2_}')

# --------------------------------------------- Crossover ---------------------------------------------

def crossover(indiv_1, indiv_2, gen):
    son1 = []
    son2 = []

    for i in range(gen):
        son1.append(indiv_1[i])
    for i in range(gen, indiv_1.shape[0]):
        son1.append(indiv_2[i])
    for i in range(gen):
        son2.append(indiv_2[i])
    for i in range(gen,indiv_2.shape[0]):
        son2.append(indiv_1[i])

    return son1,son2

son1, son2 = crossover(_max_,_max2_,2)
son3, son4 = crossover(_max_,_max2_,3)
son1 = np.array(son1)
son2 = np.array(son2)
son3 = np.array(son3)
son4 = np.array(son4)

print(f'Padres: \n{_max_}\n{_max2_}')
print(f'Hijos: \n{son1}\n{son2}\n{son3}\n{son4}')

# Armo la nueva población
new_pop = []
new_pop.append(_max_)
new_pop.append(_max2_)
new_pop.append(son1)
new_pop.append(son2)
new_pop.append(son3)
new_pop.append(son4)
new_pop = np.array(new_pop)

print(f'Nueva población: \n{new_pop}')


# ---------------------------------------------- Mutación ---------------------------------------------

def mutation(individual, gen):

    if (individual[gen] == 1):
        individual[gen] = 0
    else:
        individual[gen] = 1

    return individual

gen_mut = 2
son = 3
mutant = mutation(new_pop[2+son-1], gen_mut-1)
new_pop[2+son-1] = mutant

print(f'Población con el tercer hijo mutado en su gen {gen_mut}: \n{new_pop}')

# --------------------------------------------- Selección ---------------------------------------------

sum = 0
for i in range(new_pop.shape[0]):
    sum += calculate_fitness(new_pop[i])
    print(f'\nf(x): {calculate_fitness(new_pop[i])}')

_max_,i_max = best(new_pop)
print(f'El mejor individuo de la nueva población es: {_max_}')

print(f'El valor promedio de la f(x) es: {"%.2f" % (sum/new_pop.shape[0])}')