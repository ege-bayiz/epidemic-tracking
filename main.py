import dgl
import dgl.nn as dglnn
from dgl.data import DGLDataset
import dgl.function as dglfn

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class GATConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GATConv(
            in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = dglnn.GATConv(
            in_feats=h_feats,  out_feats=num_classes, aggregator_type='mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
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

def train(g):
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['age']
    print(features.type())
    labels = g.ndata['health']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    g.edata["time_spent"] = g.edata["time_spent"].type(dtype=torch.float64)
    g.update_all(dglfn.copy_e("time_spent", "feat_copy"), dglfn.sum("feat_copy", "feat"))
    features = g.ndata["feat"].clone().view(g.ndata["feat"].shape[0], 1).type(dtype=torch.long)   
    print(features.shape)
    print(features.type())


    model = GCN(1, 16, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(50):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
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

        if e % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

def main():
    filename = '2020-01-31.dgl'
    name = "SIR"
    dataset = Dataset(name=name, filename=filename)
    graph = dataset[0]
    
    train(graph)
if __name__ == '__main__':
    main()
