# GNN Node Classification for Epidemics
This is the demo code for the semester project of ECE 382V Complex Networks in the Real World. 

Given the co-location networks recorded in Austin, TX throughout 2020, this project creates spatio-temporal graphs, simulates an SIR model based epidemic on these networks, and then, using these graphs, Graph Neural Network based algorithms are trained to predict the health labels, i.e. if an individual is susceptible, infected or recovered.

## generate_spatiotemporal_graphs.py:

This scipt is used to generate a spatiotemporal co-location graph on which an SIR model based epidemic is simulated. Initially, the basic reproduction number for the SIR model, the length of the infection period in which an infected individual can spread the disease, the number of infected nodes to select initially at random at the beginning of the simulation, and lastly the preference for creating a spatial or spatio-temporal graph are parametrized and should be determined before running the script.

## plot_epidemic.py:

This script is used to run an epidemic simulation and plot the results. The parameters are same as "generate_spatiotemporal_graphs.py" except the last one.

## train_GNN_models.py:

This script is used train GNN models on the graphs generated in "generate_spatiotemporal_graphs.py" to predict their nodes' health labels.
