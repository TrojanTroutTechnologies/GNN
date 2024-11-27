import torch
import torch.nn as nn
import torch_scatter
import torch_geometric as pyg


class Encode_NodeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Two-hidden-layers
        self.layers = nn.Sequential(
            nn.Linear(22, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.layers(x)


class Encode_EdgeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Two-hidden-layers
        self.layers = nn.Sequential(
            nn.Linear(3, 128),  # 2D + relative distance
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.layers(x)


class Processor_NodeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Two-hidden-layers
        self.layers = nn.Sequential(
            nn.Linear(128 * 2, 128),  # 2D
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.layers(x)


class Processor_EdgeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Two-hidden-layers
        self.layers = nn.Sequential(
            nn.Linear(128 * 3, 128),  # 2D + relative distance
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder_MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Two-hidden-layers
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2D
        )

    def forward(self, x):
        return self.layers(x)


class InteractionNetwork(pyg.nn.MessagePassing):
    def __init__(self):
        super().__init__()
        self.NodeMLP = Processor_NodeMLP()
        self.EdgeMLP = Processor_EdgeMLP()

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        node_out = self.NodeMLP(torch.cat((x, aggr), dim=-1))

        # Residual add
        edge_out += edge_feature
        node_out += x

        return node_out, edge_out

    def message(self, edge_index, x_i, x_j, edge_feature):
        edge_in = torch.cat((x_i, x_j, edge_feature), dim=-1)
        edge_out = self.EdgeMLP(edge_in)
        return edge_out

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum"
        )
        return (inputs, out)


class LearnedSimulator(torch.nn.Module):
    def __init__(self, gnn_layers=10, num_particle_types=9, particle_type_size=16):
        super().__init__()
        self.gnn_layers = gnn_layers
        self.embedding = nn.Embedding(num_particle_types, particle_type_size)

        self.node_in = Encode_NodeMLP()
        self.edge_in = Encode_EdgeMLP()
        self.decoder = Decoder_MLP()
        self.layers = nn.ModuleList([InteractionNetwork() for _ in range(gnn_layers)])

    def forward(self, data):
        node_feature = torch.cat((self.embedding(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)

        for i in range(self.gnn_layers):
            node_feature, edge_feature = self.layers[i](
                node_feature, data.edge_index, edge_feature=edge_feature
            )

        out = self.decoder(node_feature)
        return out
