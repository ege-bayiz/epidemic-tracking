# This scipt is used to generate a spatiotemporal co-location graph
# on which an SIR model based epidemic is simulated

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
INFECTIOUS_PERIOD = 14 # (8 is typical for measles) Higher value for faster spread
NUM_INFECTED = 100 # Initial infected population size
NUM_DAYS = 30 # Number of days in a month to simulate
NUM_DAYS_IN_OUT = 8 # Window length (final days in the simulation) to create a dgl graph for training
SPATIO_TEMPORAL = True # True to create spatio-temporal graph, False to have each day separately

months = [4] # Months selected for SIR simulation
sir = {'S': set(),
       'I': (set(), {}),
       'R': set()}
S = []
I = []
R = []

# Categorize age groups to integers given as string values in the co-location data
FEAT_AGE = {"NaN": 0}
for i in range(20,90,5):
    key = "Age_"+str(i)+"_to_"+str(i+4)
    idx = i//5-4+1
    FEAT_AGE[key] = int(idx)

G_spatiotemp = None # Spatio temporal graph to generate
node_attr_dict = {}  # Node attribute dictionary to keep record of the attributes of all nodes
sir_spatiotemp = {'S': set(),
                    'I': (set(), {}),
                    'R': set()} # SIR dictionary

# This section loads every day's co-location network
# Renames every node with respect to the day it is on
# Generate the temporal connection if a spatio-temporal graph is to be created
print("GENERATING SUSCEPTIBLE POPULATION...")
first_day = True
for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, NUM_DAYS+1), position=1, desc="days  ", leave=False, colour='red', ncols=80):
        try:
            # Load co-location network
            filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
            G = nx.read_gexf(filename)
        except:
            continue

        sir['S'].update(G.nodes)
        node_attrs = {}
        for node_id in G.nodes:
            node_attrs[node_id] = G.nodes[node_id]
            node_attrs[node_id]["age"] = FEAT_AGE[node_attrs[node_id]["age"]]
        node_attr_dict.update(node_attrs)
        nx.set_node_attributes(G, node_attr_dict) 
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
            mapping_id = {}
            for node_id in G_new.nodes:
                new_node_id = node_id+"_"+day_+"_"+month_    
                mapping_id[node_id] = new_node_id
            G_new = nx.relabel_nodes(G_new, mapping_id)
            
            # Add new node ids to susceptibles list
            G_spatiotemp_nodes = set([node_id[:-6] for node_id in G_spatiotemp.nodes])
            for node_id in G_spatiotemp_nodes:
                new_node_id = node_id+"_"+day_+"_"+month_ 
                mapping_id[node_id] = new_node_id
                G_spatiotemp.add_node(new_node_id)
                node_attr = {new_node_id: node_attr_dict[node_id]}
                nx.set_node_attributes(G_spatiotemp, node_attr) 
           
            sir_spatiotemp['S'].update(list(mapping_id.values()))
            # Add the current modified graph to the spatiotemporal graph
            G_spatiotemp = nx.compose(G_spatiotemp,G_new)

            if SPATIO_TEMPORAL:
                # Connect a node's previous and current day ids
                # Add timespent edge attribute, set to the num of mins in a day
                for node_id in mapping_id.values():
                    node_id_ = node_id[:-6]
                    if node_id_ in prev_mapping_id:
                        prev_node_id = prev_mapping_id[node_id_]
                        G_spatiotemp.add_edge(prev_node_id, node_id, time_spent= 24*60)

        prev_mapping_id = mapping_id
        
        first_day = False
        #########################

# This section simulates SIR model based epidemic
pop_size = len(sir['S'])
print("SIMULATING DISEASE SPREAD...")
first_day = True
for month in tqdm(months, position=0, desc="months", leave=False, colour='green', ncols=80):
    for day in tqdm(range(1, NUM_DAYS+1), position=1, desc="days", leave=False, colour='red', ncols=80):
        try:
            # Load co-location network
            filename = 'Data/2020/{:02d}/2020-{:02d}-{:02d}.gexf'.format(month, month, day)
            G = nx.read_gexf(filename)
        except:
            continue


        #########################   
        # SPATIO-TEMPORAL GRAPH #
        # Dictionary to keep track of the new IDs of the nodes with respect to the day they are from
        day_ = str(day) if day//10>=1 else "0"+str(day)
        month_ = str(month) if month//10>=1 else "0"+str(month)
        mapping_id = {}
        for node_id in G.nodes:
            mapping_id[node_id] = node_id+"_"+day_+"_"+month_      
        #########################   

        if first_day:
            #Compute spreadiong constant rho
            degrees = np.zeros(len(G.nodes))
            idx = 0
            for i in G.nodes:
                for j in nx.neighbors(G, i):
                    degrees[idx] += G[i][j]['time_spent']
                idx += 1
            rho = r0 / (INFECTIOUS_PERIOD * np.mean(degrees))

            # Randomly sample NUM_INFECTED many nodes to start the spread
            # Remove these from the set of susceptible nodes and add to the set of infected ones
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

        #########################
        # SPATIO-TEMPORAL GRAPH #
        # Remove recovered nodes from the set of susceptible ones
        for i in sir['R']:
            new_node_id = i+"_"+day_+"_"+month_   
            if new_node_id in sir_spatiotemp['S']:
                sir_spatiotemp['S'].remove(new_node_id)
                sir_spatiotemp['R'].add(new_node_id)
        #########################

        # Remove infected nodes from the set of susceptible nodes
        temp_infected = sir['I'][0].copy()
        for i in temp_infected:
            sir['I'][1][i] += 1
            
            #########################
            # SPATIO-TEMPORAL GRAPH #
            new_node_id = i+"_"+day_+"_"+month_   
            if not first_day:
                sir_spatiotemp['S'].remove(new_node_id)
                sir_spatiotemp['I'][0].add(new_node_id)
                sir_spatiotemp['I'][1][new_node_id] = sir['I'][1][i] 
            #########################

            # If a node recover, put it into the set of recovered nodes
            day = sir['I'][1][i]
            if day >= INFECTIOUS_PERIOD:
                sir['I'][0].remove(i)
                sir['I'][1].pop(i)
                sir['R'].add(i)

                #########################
                # SPATIO-TEMPORAL GRAPH #
                sir_spatiotemp['I'][0].remove(new_node_id)
                sir_spatiotemp['I'][1].pop(new_node_id)
                sir_spatiotemp['R'].add(new_node_id)
                #########################
                continue

            # If the infected node is seen in this day's graph
            # Simulate infection events with respect to time
            # that the node spends with its neighbors
            if i in G.nodes:
                for j in nx.neighbors(G, i):
                    if j in sir['S']:
                        D = G[i][j]['time_spent']
                        # If an infection occurs, add the neighbor to the set of infected nodes
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
# Using the result of the SIR spread
# Add health labels to each node
attrs = {}
for n in G_spatiotemp.nodes:
    attr = G_spatiotemp.nodes[n]
    if n in sir_spatiotemp['S']:
        attr["health"] = 0
    elif n in sir_spatiotemp['I'][0]:
        attr["health"] = 1
    elif n in sir_spatiotemp['R']:
        attr["health"] = 2
    else:
        raise ValueError("WHERE IS THIS NODE??")
    attrs[n] = attr
nx.set_node_attributes(G_spatiotemp, attrs)

# Remove isolated nodes
G_spatiotemp.remove_nodes_from(list(nx.isolates(G_spatiotemp)))

# Convert the graph into a directed one
G_spatiotemp = G_spatiotemp.to_directed()

# Filter the graph to save with respect to the selected window length
nodes = []
for n in G_spatiotemp.nodes:
    day = n[-2:]
    mon = n[-5:-3]
    if int(n[-2:]) == months[-1] and int(n[-5:-3]) >= NUM_DAYS - NUM_DAYS_IN_OUT:
        nodes.append(n)
G_spatiotemp = nx.subgraph(G_spatiotemp, nodes)

# Save graph in gexf format
filename = '2020'
for month in months:
    month_ = str(month) if month//10>1 else "0"+str(month)
    filename += "-"+str(month_)
if SPATIO_TEMPORAL:
    filename_gexf = filename+"_spatiotemp.gexf"
else:
    filename_gexf = filename+".gexf"
nx.write_gexf(G_spatiotemp,filename_gexf)

# Save graph in dgl format
G_dgl = dgl.from_networkx(G_spatiotemp, node_attrs=['health', 'age'], edge_attrs=['time_spent'])
if SPATIO_TEMPORAL:
    filename_dgl = filename+"_spatiotemp.dgl"
else:
    filename_dgl = filename+".dgl"
dgl.save_graphs(filename_dgl, G_dgl)
#########################


