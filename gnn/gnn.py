# This is an implementation of the life-long learning ideas When the graph get larger, it does not fit memory. Also
# to build FM we need to support adding database incrementally. New database may have a small overlap with the
# existing database and we don't know this in advance. New database may have new schema with new data
# properties/modalities/entities.
# Implementation of graph
# auto-scaled Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec: GNNAutoScale: Scalable and Expressive
# Graph Neural Networks via Historical Embeddings (ICML 2021) for GPU-memory efficient GNN training.
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from gnn.classifier_head import ClassifierHead
from gnn.distmult import DistMult
from gnn.node import Node
from gnn.transe import TransE
from gnn.utils import clean_modality_name


class LinkPredictor(torch.nn.Module):
    """A class implementing a link prediction with WRGCN encoder and DistMult decoder"""

    def __init__(self, graph_cfg, gnn_cfg, scoring_type="DistMult", device='cpu'):
        super(LinkPredictor, self).__init__()
        self.gnn_cfg = gnn_cfg
        self.num_rel = graph_cfg.get_num_relations()
        self.h_dim = gnn_cfg['h_dim']
        self.num_hidden_layers = gnn_cfg['num_hidden_layers']
        self.output_dim = gnn_cfg['output_dim']
        self.node_projection_embedding_size = self.gnn_cfg['node_projection_embedding_size']

        # node projection, for each type of node, e.g. Protein_Sequence, there is a projection layer for that type
        node_projections = {}
        for modality in graph_cfg.modalities:
            node_projections[clean_modality_name(modality)] = Node(modality,
                                                                   graph_cfg.modalities[modality]['projection'],
                                                                   graph_cfg.modalities[modality]['embedding_size'],
                                                                   self.node_projection_embedding_size)
        self.node_projections = torch.nn.ModuleDict(node_projections)

        # graph conv layer project node embeddings to output embeddings
        self.graph_conv = []
        self.graph_conv.append(RGCNConv(self.node_projection_embedding_size, self.h_dim, self.num_rel))
        for i in range(self.num_hidden_layers):
            self.graph_conv.append(RGCNConv(self.h_dim, self.h_dim, self.num_rel))
        self.graph_conv.append(RGCNConv(self.h_dim, self.output_dim, self.num_rel))
        self.graph_conv = torch.nn.ModuleList(self.graph_conv)
        # scoring layer
        if scoring_type == "DistMult":
            self.scoring = DistMult(self.num_rel, self.output_dim, device=self.device)
        elif scoring_type == "TransE":
            self.scoring = TransE(self.num_rel, self.output_dim, device=self.device)
        else:
            self.scoring = ClassifierHead(self.num_rel, self.output_dim, device=self.device)

        self.to(self.device)

    def projection(self, nodes):
        projected_embeddings = []
        node_indices = []
        for node_type, node_data in nodes.items():
            x = self.node_projections[clean_modality_name(node_type)](node_data['embeddings'])
            projected_embeddings.append(x)
            node_indices.append(node_data['node_indices'])
        node_indices = torch.cat(node_indices)
        projected_embeddings = torch.cat(projected_embeddings)
        projected_embeddings = [(i, e) for i, e in zip(node_indices, projected_embeddings)]
        projected_embeddings.sort(key=lambda x: x[0])
        projected_embeddings = [x[1].unsqueeze(0) for x in projected_embeddings]
        projected_embeddings = torch.cat(projected_embeddings)
        return projected_embeddings

    def encoder(self, nodes, triples: torch.Tensor):
        e_mask, r_mask = [True, False, True], [False, True, False]
        # projection
        node_projection_embeddings = self.projection(nodes)
        # convolution
        u = node_projection_embeddings
        for graph_conv in self.graph_conv[:-1]:
            u = F.relu(graph_conv(u, triples[e_mask, :], triples[r_mask, :][0]))
        u = self.graph_conv[-1](u, triples[e_mask, :], triples[r_mask, :][0])
        return u

    def decoder(self, out_emb, triples):
        return self.scoring(out_emb, triples)

    def forward(self, nodes, triples: torch.Tensor,
                neg_triples: Optional[torch.Tensor] = None):
        output = self.encoder(nodes, triples)
        if neg_triples is not None:
            triples = torch.cat((triples, neg_triples), -1)

        scores = self.decoder(output, triples)
        return output, scores
