# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:09:15 2019

@author: dominique
"""

###############################################################################
### Import
###############################################################################

import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import matplotlib.pyplot as plt
import pandas as pd

# for reproduction
s = None
random.seed(s)
np.random.seed(s)

###############################################################################
### Functions
###############################################################################

def fObj_test(individual):
    f = toolbox.compile(individual)
    Yp = f(df.Pi1,df.Pi2)
    #Yp = f(df.Pi1,df.Pi7,df.Pi8)
    Serr_log = (10*np.log10(df.PiF)- 10*np.log10(Yp))**2
    Serr_MSE = (df.PiF- Yp)**2
    
    #compute range
    range1 = (np.max(df.PiF)-np.min(df.PiF))**2
    range2 = (np.max(10*np.log10(df.PiF))-np.min(10*np.log10(df.PiF)))**2
    
    if np.isnan(Serr_log).any():
        MSE = 1e99
    elif np.isnan(Serr_MSE).any():
        MSE = 1e99   
    else:
        MSE = 100*np.sqrt((0.5*np.mean(Serr_log)/range2 + 0.5*np.mean(Serr_MSE)/range1))
        #MSE = 100*np.mean(Serr_MSE/range1)
    return MSE,


def fObj_MSE(individual):
    f = toolbox.compile(individual)
    Yp = f(df.Pi1,df.Pi2)
    Serr = (df.PiF- Yp)**2
    MSE = np.mean(Serr)
    if np.isnan(Serr).any():
        MSE = 1e99
    else:
        MSE = np.mean(Serr)
    return MSE,
    

def linker_goody(x1, x2, x3): 
    return operator.truediv(x1,(x2+x3))           

def protected_div(x1, x2):
    if np.isscalar(x2):
        if abs(x2) < 1e-6:
            x2 = float('nan')
    else:
        x2[x2<1e-6] = float('nan')
    return x1 / x2

def root(x1):
    return np.sqrt(x1)

def epow(x1):
    return np.exp(x1)

###############################################################################
### Syntetic dataset of Goody
###############################################################################

from Dataset import Goody_data

f = np.logspace(-3.5,2,num=2000)
Rt = np.logspace(0,2.5,num = 5)
df = Goody_data(f,Rt)

###############################################################################
### GEP
###############################################################################

# Generate the fset of function and terminals

pset = gep.PrimitiveSet('Main', input_names=['X','Y'])
pset.add_function(operator.add, 2)
pset.add_function(operator.mul, 2)
pset.add_function(operator.truediv, 2)
pset.add_rnc_terminal()
pset.add_pow_terminal('X') #attention: Must the same as input in primitive set
pset.add_pow_terminal('Y')

pset.add_constant_terminal(1.0)

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 4 # head length
n_genes = 3   # number of genes in a chromosome
r = h*2 + 1   # length of the RNC array

# size of population and number of generations
n_pop = 100
n_gen = 1001
N_eval = 1


toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.choice, np.arange(0.1,10.0,0.1))
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=linker_goody)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

toolbox.register('evaluate', fObj_test)

toolbox.register('select', tools.selTournament, k=round((2.0/3.0)*n_pop)+1 , tournsize=2)
#toolbox.register('select', tools.selTournament, k=n_pop , tournsize=2)

# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb='4p', pb=0.1)
#toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.025)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.025)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.025)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.05)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.05)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.025)
# 2. Dc-specific operators
#toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb='4p', pb=0.8)
#toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
#toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='4p', pb=0.1)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

Max_evolution = empty_lists = [ [] for i in range(N_eval) ]
Hof_save = empty_lists = [ [] for i in range(N_eval) ]
Max_fit = np.zeros(N_eval)

for i in range(N_eval):
    # start evolution
    
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)   # only record the best three individuals ever found in all generations
    #hof = tools.ParetoFront(100)
    
    pop, log = gep.gep_simple_opt(pop, toolbox, n_generations=n_gen, 
                                  n_elites=1, stats=stats, 
                                  hall_of_fame=hof, verbose=True, 
                                  optimizer= True, opt_period=1, opt_prob=0.005, opt_bounds=(0.001,10))
    Max_evolution[i] = log.select("min")
    Hof_save[i] = hof


print(hof[0])

##print best indiviudal and simplify
best_ind = Hof_save[0][0]
best_func = toolbox.compile(best_ind)
print(best_ind)

#display
Pi1_test = np.linspace(np.min(df.Pi1),np.max(df.Pi1),num=10000)
plt.figure()
plt.semilogx(df.Pi1,10*np.log10(df.PiF),color='b',marker='.',linestyle=' ')
for ii in np.unique(df.Pi2):
    plt.semilogx(Pi1_test,10*np.log10(best_func(Pi1_test,ii)))
plt.show()


# convergence
plt.figure()
plt.plot(Max_evolution[0])

temp = np.zeros(N_eval)
for i in range(N_eval):
    print(Hof_save[i][0])
    temp[i] = Max_evolution[i][-1]
