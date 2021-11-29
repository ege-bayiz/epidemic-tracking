import dgl
import dgl.nn as dglnn
from dgl.data import DGLDataset
import dgl.function as dglfn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

USE_EDGE_FEATURES = True
USE_NODE_FEATURES = True

class Dataset(DGLDataset):
    def __init__(self, name, filename):
        self.filename = filename
        super().__init__(name=name)
        
    def process(self):
        (self.graph,), _ = dgl.load_graphs(self.filename)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.graph.num_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        pop = [*range(n_nodes)]
        train_idx = random.sample(pop,n_train)
        for train_i in train_idx:
            pop.remove(train_i)
        val_idx = random.sample(pop,n_val)
        for val_i in val_idx:
            pop.remove(val_i)
        test_idx = pop
        # train_mask[:n_train] = True
        # val_mask[n_train:n_train + n_val] = True
        # test_mask[n_train + n_val:] = True
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class GATConv(nn.Module):
    def __init__(self, in_feats, h_feats, e_feats, e_emb, num_classes):
        super(GATConv, self).__init__()
        NUM_HEADS = 2
        self.edge_encoder = nn.Linear(in_features=e_feats, out_features=e_emb)
        self.conv1 = dglnn.GATConv(
            in_feats=in_feats, out_feats=h_feats, num_heads=NUM_HEADS)
        self.inter = nn.Linear(in_features=NUM_HEADS*h_feats, out_features=h_feats)
        self.conv2 = dglnn.GATConv(
            in_feats=h_feats,  out_feats=num_classes, num_heads=1)


    def forward(self, g, in_feat, e_feat):
        e_feat_emb = self.edge_encoder(e_feat)
        e_feat_emb = F.relu(e_feat_emb, inplace=True)
        g.edata["e_feat_emb"] = e_feat_emb
        g.update_all(dglfn.copy_e("e_feat_emb", "e_feat_emb_copy"), dglfn.sum("e_feat_emb_copy", "n_e_feat_emb"))
        features_emb = g.ndata["n_e_feat_emb"].clone().view(g.ndata["n_e_feat_emb"].shape[0], 1)
        features_concat = torch.cat([in_feat, features_emb], -1)
        h = self.conv1(g, features_concat)
        h = F.relu(h)
        h = h.permute(1, 0, 2).reshape((h.size(dim=0), h.size(dim=1)*h.size(dim=2)))
        h = self.inter(h)
        h = self.conv2(g, h)
        h = h.permute(1, 0, 2).reshape((h.size(dim=0), h.size(dim=1)*h.size(dim=2)))
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=h_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats)
        self.conv2 = dglnn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, date):
    # algos = [GCN, SAGE, GATConv]
    # algo_names = ['graph-conv', 'graph-SAGE', 'GAT-Conv']
    # algos = [GCN]
    # algo_names = ['graph-conv']
    algos = [GATConv]
    algo_names = ['GAT-Conv']
    results = pd.DataFrame({'epoch': [],
                            'loss': [],
                            'algorithm': [],
                            'date': [],
                            'trial': [],
                            'train_acc': [],
                            'validation_acc': [],
                            'test_acc': []})
    # print(features.type())
    labels = g.ndata['health']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    # Type 1: Use age features
    g.ndata['age'] = g.ndata['age'].type(dtype=torch.float32).view(g.ndata['age'].shape[0], 1)
    features_age= g.ndata['age'].clone()
    print(features_age.shape)
    print(features_age.type())
    # Type 2: Use aggregated timespent features
    g.edata["time_spent"] = g.edata["time_spent"].type(dtype=torch.float32)
    g.update_all(dglfn.copy_e("time_spent", "feat_copy"), dglfn.sum("feat_copy", "feat"))
    g.ndata["feat"] = g.ndata["feat"].view(g.ndata["feat"].shape[0], 1)
    features_agg_ts = g.ndata["feat"].clone()
    print(features_agg_ts.shape)
    print(features_agg_ts.type())
    # Type 3: Use age and aggregated timespent features by concating them
    g.apply_nodes(lambda nodes: {'concat': torch.cat([nodes.data['age'], nodes.data["feat"]], -1)})
    features_concat = g.ndata["concat"].clone()
    print(features_concat.shape)
    print(features_concat.type())
    # Type 4: Use age and timespent features separately to encode edge features in forward pass and then concat
    g.edata["time_spent"] = g.edata["time_spent"].view(g.edata["time_spent"].shape[0], 1)
    features_edge = g.edata["time_spent"].clone()
    print(features_edge.shape)
    print(features_edge.type())

    for trial in range(10):
        for k in range(len(algos)):
            best_val_acc = 0
            best_test_acc = 0

            if algo_names[k] == 'GAT-Conv':
                model = algos[k](2, 16, 1, 1, 3)
            else:
                model = algos[k](1, 16, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for e in range(100):
                # Forward
                if algo_names[k] == 'GAT-Conv':
                    logits = model(g, features_age, features_edge)
                else:
                    logits = model(g, features_agg_ts)

                # Compute prediction
                pred = logits.argmax(-1)
                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                if algo_names[k] == 'GAT-Conv':
                    logp = F.log_softmax(logits, 1)
                    loss = F.nll_loss(logp[train_mask], labels[train_mask])
                else:
                    loss = F.cross_entropy(logits[train_mask], labels[train_mask])

                # Compute accuracy on training/validation/test
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

                # Save the best validation accuracy and the corresponding test accuracy.
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if e % 99 == 0:
                    print('Trial: {} - In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                        trial, e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

                results = results.append({'epoch': e,
                                'loss': loss.item(),
                                'algorithm': algo_names[k],
                                'date': date,
                                'trial': trial,
                                'train_acc': train_acc.item(),
                                'validation_acc': val_acc.item(),
                                'test_acc': test_acc.item()}, ignore_index=True)
    return results

def plot(results):
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=results, x="epoch", y="train_acc", hue="algorithm")
    plt.title("Training Accuracy")

    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=results, x="epoch", y="test_acc", hue="algorithm")
    plt.title("Test Accuracy")

    df = results.query("algorithm == 'graph-conv'")
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x="epoch", y="train_acc", hue="date")
    plt.title("Training Accuracy of GCN")

    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x="epoch", y="test_acc", hue="date")
    plt.title("Test Accuracy of GCN")
    plt.show()

def main():
    filename = '2020-05_spatiotemp.dgl'
    date = "2020-05"
    name = "SIR"
    dataset = Dataset(name=name, filename=filename)
    graph = dataset[0]
    results = train(graph, date)

    # filename = '2020-01-31.dgl'
    # date = "2020-01-31"
    # name = "SIR"
    # dataset = Dataset(name=name, filename=filename)
    # graph = dataset[0]
    # results1 = train(graph, date)

    # filename = '2020-03-31.dgl'
    # date = "2020-03-31"
    # name = "SIR"
    # dataset = Dataset(name=name, filename=filename)
    # graph = dataset[0]
    # results2 = train(graph, date)

    # filename = '2020-07-31.dgl'
    # date = "2020-07-31"
    # name = "SIR"
    # dataset = Dataset(name=name, filename=filename)
    # graph = dataset[0]
    # results3 = train(graph, date)

    # results = results1.append(results2)
    # results = results.append(results3)
    # results.reset_index(inplace=True)
    results.to_csv('convergence_results_new.csv', index=False)
    # print(results.head())
    plot(results)


if __name__ == '__main__':
    main()
