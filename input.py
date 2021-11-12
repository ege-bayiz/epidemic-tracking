import networkx as nx
import random as rng
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stat
from enum import Enum
import dgl

r0 = 16 # Basic reproduction number (16 is typical for measles)
INFECTIOUS_PERIOD = 8 # (8 is typical for measles)
NUM_INFECTED = 100 # Initial infected population size
NUM_DAYS = 31

months = [1]
sir = {'S': set(),
       'I': (set(), {}),
       'R': set()}
S = []
I = []
R = []

FEAT_AGE = {"NaN": [0 for i in range(14)]}
for i in range(20,90,5):
    key = "Age_"+str(i)+"_to_"+str(i+4)
    feat = [0 for i in range(14)]
    idx = i//5-4
    feat[idx] = 1
    FEAT_AGE[key] = feat

print("GENERATING SUSCEPTIBLE POPULATION...")
for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, NUM_DAYS+1), position=1, desc="days  ", leave=False, colour='red', ncols=80):
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
    for day in tqdm(range(1, NUM_DAYS+1), position=1, desc="days", leave=False, colour='red', ncols=80):
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

filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(months[0], months[0], NUM_DAYS)
G = nx.read_gexf(filename)
attrs = {}
ages = nx.get_node_attributes(G,'age')
for n in G.nodes:
    feat_age = FEAT_AGE[ages[n]]
    if n in sir['S']:
        attrs[n] = {"health": 0,"age":feat_age}
    elif n in sir['I'][0]:
        attrs[n] = {"health": 1,"age":feat_age}    
    elif n in sir['R']:
        attrs[n] = {"health": 2,"age":feat_age}
    else:
        raise ValueError("WHERE IS THIS NODE??")
        
nx.set_node_attributes(G, attrs)
G = G.to_directed()
G_dgl = dgl.from_networkx(G, node_attrs=['health', 'age'], edge_attrs=['time_spent'])
filename = '2020-{:02d}-{:02d}.dgl'.format(months[0], NUM_DAYS)
dgl.save_graphs(filename, G_dgl)


