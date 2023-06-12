import torch.nn as nn
import torch.nn.functional as F
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class BindingAffinitySeparate(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_drug, n_target):
        super(BindingAffinitySeparate, self).__init__()
        hd1 = 1024
        hd2 = 512
        self.drug1 = nn.Linear(n_drug, hd1)
        self.drug2 = nn.Linear(hd1, hd2)
        self.target1 = nn.Linear(n_target, hd1)
        self.target2 = nn.Linear(hd1, hd2)
        self.combine1 = nn.Linear(2*hd2, hd2)
        self.combine2 = nn.Linear(hd2, 1)
        self.to(device)

    def forward(self, d, t):
        x = F.relu(self.drug1(d))
        x = self.drug2(x)
        y = F.relu(self.target1(t))
        y = self.target2(y)
        c = F.relu(self.combine1(torch.cat([x, y], dim=-1)))
        c = self.combine2(c)
        return c


class BindingAffinity(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_drug, n_target, gnn_embedding_dim=256):
        super(BindingAffinity, self).__init__()
        hd1 = 1024
        hd2 = 512
        self.gnn_embedding_dim = gnn_embedding_dim
        self.combine1 = nn.Linear(n_drug+n_target-2*gnn_embedding_dim, hd1)
        self.combine2 = nn.Linear(hd1, hd2)
        self.combine3 = nn.Linear(hd2, 1)
        self.know1 = nn.Linear(2*gnn_embedding_dim, 1024)
        self.know2 = nn.Linear(1024, 1024)
        self.know3 = nn.Linear(1024, 1)
        self.to(device)

    def forward(self, d, t):
        gnn_d = d[:, :self.gnn_embedding_dim]
        gnn_t = t[:, :self.gnn_embedding_dim]
        c = F.relu(self.combine1(torch.cat([d[:, self.gnn_embedding_dim:], t[:, self.gnn_embedding_dim:]], dim=-1)))
        c = F.relu(self.combine2(c))
        c = self.combine3(c)

        g = F.relu(self.know1(torch.cat([gnn_d, gnn_t], dim=-1)))
        g = F.relu(self.know2(g))
        g = self.know3(g)
        return g + c


class BindingAffinityInitial(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_drug, n_target):
        super(BindingAffinityInitial, self).__init__()
        hd1 = 1024
        hd2 = 512
        self.combine1 = nn.Linear(n_drug+n_target, hd1)
        self.combine2 = nn.Linear(hd1, hd2)
        self.combine3 = nn.Linear(hd2, 1)
        self.to(device)

    def forward(self, d, t):
        c = F.relu(self.combine1(torch.cat([d, t], dim=-1)))
        c = F.relu(self.combine2(c))
        c = self.combine3(c)
        return c
