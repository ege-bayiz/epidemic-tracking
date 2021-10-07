import networkx as nx
import random as rng
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stat
from enum import Enum

#Reading graph data
# day = 5
# month = 3
# filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month,month,day)
#
# G1 = nx.read_gexf(filename)
# day = 6
#
# filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month,month,day)
# G2 = nx.read_gexf(filename)
# print(len(G1.nodes))
# print(len(G2.nodes))
# print(sorted(G1.nodes) == sorted(G2.nodes))

r0 = 2.2  # epidemic constant
rho = 0.0005  # spreading constant (per minute)
INFECTIOUS_PERIOD = 10

sir = {'S': set(),
       'I': (set(), {}),
       'R': set()}
S = []
I = []
R = []
for month in [3]:
    print('Month : ', month)
    for day in tqdm(range(1,32)):
        filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
        G = nx.read_gexf(filename)
        #print('')
        #print(str(len(sir['S'])) + ' : ' + str(len(sir['I'][0])) + ' : ' + str(len(sir['R'])))
        test1 = len(G.nodes)
        count = 0
        for i in G.nodes:
            if day == 1 and count < 10:
                sir['I'][0].add(i)
                sir['I'][1][i] = 0
            elif not(i in sir['I'][0] or i in sir['R']):
                sir['S'].add(i)

            count += 1

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
                        if rng.random() > 1-(1-rho)**D:
                            sir['S'].remove(j)
                            sir['I'][0].add(j)
                            sir['I'][1][j] = 0

        S.append(len(sir['S']))
        I.append(len(sir['I'][0]))
        R.append(len(sir['R']))

plt.figure(figsize=(8, 6))
plt.plot(range(len(S)), S, color='r', label='S')
plt.plot(range(len(I)), I, color='g', label='I')
plt.plot(range(len(R)), R, color='b', label='R')
plt.legend()
plt.show()







