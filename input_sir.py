import networkx as nx
import random as rng
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stat
from enum import Enum
import dgl
import copy
from pathlib import Path

r0 = 16 # Basic reproduction number (16 is typical for measles)
INFECTIOUS_PERIOD = 8 # (8 is typical for measles)
NUM_INFECTED = 100 # Initial infected population size
NUM_DAYS = 8

months = [5]
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

G_spatiotemp = None
sir_spatiotemp = {'S': set(),
                    'I': (set(), {}),
                    'R': set()}
print("GENERATING SUSCEPTIBLE POPULATION...")
first_day = True
for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, NUM_DAYS+1), position=1, desc="days  ", leave=False, colour='red', ncols=80):
        try:
            filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
            G = nx.read_gexf(filename)
        except:
            continue

        sir['S'].update(G.nodes)

        #########################
        # SPATIO-TEMPORAL GRAPH #
        day_ = str(day) if day//10>=1 else "0"+str(day)
        month_ = str(month) if month//10>=1 else "0"+str(month)
        # First day
        if first_day:
            G_spatiotemp = copy.deepcopy(G)
            # Rename node ids wrt month and day
            mapping_id = {}
            for node_id in G_spatiotemp.nodes:
                mapping_id[node_id] = node_id+"_"+day_+"_"+month_ 
            G_spatiotemp = nx.relabel_nodes(G_spatiotemp, mapping_id)
            # Add new node ids to susceptibles list
            sir_spatiotemp['S'].update(G_spatiotemp.nodes)
        else:
            G_new = copy.deepcopy(G)
            # Rename node id wrt month and day
            todays_nodes = set()
            mapping_id = {}
            for node_id in G_new.nodes:
                new_node_id = node_id+"_"+day_+"_"+month_    
                mapping_id[node_id] = new_node_id
                todays_nodes.add(new_node_id)
   
            G_new = nx.relabel_nodes(G_new, mapping_id)
            # Add new node ids to susceptibles list

            new_nodes = set()
            G_spatiotemp_nodes_copy = copy.deepcopy(G_spatiotemp.nodes)

            G_spatiotemp_nodes = set([node_id[:-6] for node_id in G_spatiotemp_nodes_copy])
            for node_id in G_spatiotemp_nodes:
                new_node_id = node_id+"_"+day_+"_"+month_ 
                todays_nodes.add(new_node_id)
                mapping_id[node_id] = new_node_id
                G_spatiotemp.add_node(new_node_id)
                attr = {new_node_id:G_spatiotemp.nodes[prev_mapping_id[node_id]]}
                nx.set_node_attributes(G_spatiotemp, attr) #, age=G_spatiotemp.nodes[prev_mapping_id[node_id]]["age"])  
            
            sir_spatiotemp['S'].update(list(todays_nodes))
            # Add the current modified graph to the spatiotemporal graph
            G_spatiotemp = nx.compose(G_spatiotemp,G_new)
            # Connect a node's previous and current day ids
            # Add timespent edge attribute, set to the num of mins in a day
            for node_id in todays_nodes:
                # if node_id in prev_mapping_id.keys():
                node_id_ = node_id[:-6]
                if node_id_ in prev_mapping_id:
                    prev_node_id = prev_mapping_id[node_id_]
                    G_spatiotemp.add_edge(prev_node_id, node_id)
                    edge_attr = {(prev_node_id, node_id): {"time_spent": 24*60}}
                    nx.set_edge_attributes(G_spatiotemp, edge_attr)

        prev_mapping_id = mapping_id
        first_day = False
        #########################

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


        #########################   
        # SPATIO-TEMPORAL GRAPH #
        day_ = str(day) if day//10>=1 else "0"+str(day)
        month_ = str(month) if month//10>=1 else "0"+str(month)
        mapping_id = {}
        for node_id in G.nodes:
            mapping_id[node_id] = node_id+"_"+day_+"_"+month_      
        #########################   

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

                #########################
                # SPATIO-TEMPORAL GRAPH #
                new_node_id = mapping_id[i]
                sir_spatiotemp['S'].remove(new_node_id)
                sir_spatiotemp['I'][0].add(new_node_id)
                sir_spatiotemp['I'][1][i] = 1
                #########################

        for i in sir['R']:
            new_node_id = i+"_"+day_+"_"+month_   
            if new_node_id in sir_spatiotemp['S']:
                sir_spatiotemp['S'].remove(new_node_id)
                sir_spatiotemp['R'].add(new_node_id)
            
        temp = sir['I'][0].copy()
        for i in temp:
            sir['I'][1][i] += 1
            
            #########################
            # SPATIO-TEMPORAL GRAPH #
            if not first_day:
                new_node_id = i+"_"+day_+"_"+month_   
                sir_spatiotemp['S'].remove(new_node_id)
                sir_spatiotemp['I'][0].add(new_node_id)
                sir_spatiotemp['I'][1][new_node_id] = sir['I'][1][i] 
            #########################

            day = sir['I'][1][i]
            if day >= INFECTIOUS_PERIOD:
                sir['I'][0].remove(i)
                sir['I'][1].pop(i)
                sir['R'].add(i)

                #########################
                # SPATIO-TEMPORAL GRAPH #
                sir_spatiotemp['I'][0].remove(new_node_id)
                sir_spatiotemp['I'][1].pop(i)
                sir_spatiotemp['R'].add(new_node_id)
                #########################
                continue

            if i in G.nodes:
                for j in nx.neighbors(G, i):
                    if j in sir['S']:
                        D = G[i][j]['time_spent']
                        if rng.random() < 1 - (1-rho)**D:
                            sir['S'].remove(j)
                            sir['I'][0].add(j)
                            sir['I'][1][j] = 0

                            #########################
                            # SPATIO-TEMPORAL GRAPH #
                            new_node_id_neighbor = j+"_"+day_+"_"+month_  
                            sir_spatiotemp['S'].remove(new_node_id_neighbor)
                            sir_spatiotemp['I'][0].add(new_node_id_neighbor)
                            sir_spatiotemp['I'][1][new_node_id_neighbor] = 0
                            #########################

        first_day = False
        #########################   
        # SPATIO-TEMPORAL GRAPH #
        prev_mapping_id = copy.deepcopy(mapping_id)
        #########################   

#########################   
# SPATIO-TEMPORAL GRAPH #
attrs = {}
ages = nx.get_node_attributes(G_spatiotemp,'age')
for n in G_spatiotemp.nodes:
    feat_age = FEAT_AGE[ages[n]]
    if n in sir_spatiotemp['S']:
        attrs[n] = {"health": 0,"age":feat_age}
    elif n in sir_spatiotemp['I'][0]:
        attrs[n] = {"health": 1,"age":feat_age}    
    elif n in sir_spatiotemp['R']:
        attrs[n] = {"health": 2,"age":feat_age}
    else:
        raise ValueError("WHERE IS THIS NODE??")
nx.set_node_attributes(G_spatiotemp, attrs)
G_spatiotemp = G_spatiotemp.to_directed()
filename = '2020'
for month in months:
    month_ = str(month) if month//10>1 else "0"+str(month)
    filename += "-"+str(month_)
filename_gexf = filename+".gexf"
nx.write_gexf(G_spatiotemp,filename_gexf)
G_dgl = dgl.from_networkx(G_spatiotemp, node_attrs=['health', 'age'], edge_attrs=['time_spent'])
filename_dgl = filename+".dgl"
dgl.save_graphs(filename_dgl, G_dgl)
#########################   


# filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(months[0], months[0], NUM_DAYS)
# G = nx.read_gexf(filename)
# attrs = {}
# ages = nx.get_node_attributes(G,'age')
# for n in G.nodes:
#     feat_age = FEAT_AGE[ages[n]]
#     if n in sir['S']:
#         attrs[n] = {"health": 0,"age":feat_age}
#     elif n in sir['I'][0]:
#         attrs[n] = {"health": 1,"age":feat_age}    
#     elif n in sir['R']:
#         attrs[n] = {"health": 2,"age":feat_age}
#     else:
#         raise ValueError("WHERE IS THIS NODE??")
        
# nx.set_node_attributes(G, attrs)
# G = G.to_directed()
# filename = '2020-{:02d}-{:02d}.gexf'.format(months[0], NUM_DAYS)
# nx.write_gexf(G,filename)
# G_dgl = dgl.from_networkx(G, node_attrs=['health', 'age'], edge_attrs=['time_spent'])
# filename = '2020-{:02d}-{:02d}.dgl'.format(months[0], NUM_DAYS)
# dgl.save_graphs(filename, G_dgl)


