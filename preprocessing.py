import networkx as nx
import random as rng
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stat
from enum import Enum

r0 = 16 # Basic reproduction number (16 is typical for measles)
INFECTIOUS_PERIOD = 8 # (8 is typical for measles)
NUM_INFECTED = 100 # Initial infected population size

months = [1]
sir = {'S': set(),
       'I': (set(), {}),
       'R': set()}
S = []
I = []
R = []

print("GENERATING SUSCEPTIBLE POPULATION...")
for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, 32), position=1, desc="days  ", leave=False, colour='red', ncols=80):
        try:
            filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
            G = nx.read_gexf(filename)
        except:
            continue
        sir['S'].update(G.nodes)

pop_size = len(sir['S'])

print("SIMULATING DISEASE SPREAD...")
first_day = True

for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, 32), position=1, desc="days", leave=False, colour='red', ncols=80):
        try:
            filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
            G = nx.read_gexf(filename)
        except:
            continue

        if first_day:
            #Compute spreadiong constant
            degrees = np.zeros(len(G.nodes))
            idx = 0
            for i in G.nodes:
                for j in nx.neighbors(G, i):
                    degrees[idx] += G[i][j]['time_spent']
                idx += 1

            rho = r0 / (INFECTIOUS_PERIOD * np.mean(degrees))

            infected_init = rng.sample(G.nodes, NUM_INFECTED)
            for i in infected_init:
                sir['S'].remove(i)
                sir['I'][0].add(i)
                sir['I'][1][i] = 0
            first_day = False

        temp = sir['I'][0].copy()
        for i in temp:
            sir['I'][1][i] += 1
            day = sir['I'][1][i]
            if day >= INFECTIOUS_PERIOD:
                sir['I'][0].remove(i)
                sir['I'][1].pop(i)
                sir['R'].add(i)
                continue
            if i in G.nodes:
                for j in nx.neighbors(G, i):
                    if j in sir['S']:
                        D = G[i][j]['time_spent']
                        if rng.random() < 1 - (1-rho)**D:
                            sir['S'].remove(j)
                            sir['I'][0].add(j)
                            sir['I'][1][j] = 0

        S.append(len(sir['S']))
        I.append(len(sir['I'][0]))
        R.append(len(sir['R']))

S = np.asarray(S)
I = np.asarray(I)
R = np.asarray(R)
print("PLOTTING RESULTS...")
plt.figure(figsize=(8, 6))
plt.plot(range(np.size(S)), S, color='green', label='Susceptible')
plt.plot(range(np.size(I)), I, color='red', label='Infected')
plt.plot(range(np.size(R)), R, color='grey', label='Removed')
plt.title('Simulated SIR Model')
plt.xlabel('days')
plt.ylabel('population')
plt.legend()
plt.grid()

plt.figure(figsize=(8, 6))
plt.stackplot(range(np.size(S)), S/pop_size, I/pop_size, R/pop_size,
              colors=['green', 'red', 'grey'], alpha=0.8,
              labels=['Susceptible', 'Infected', 'Removed'])
plt.title('Simulated SIR Model (Normalized, Stacked)')
plt.xlabel('days')
plt.ylabel('ratio')
plt.legend()
plt.grid()
plt.show()








