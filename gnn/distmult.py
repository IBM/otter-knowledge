import torch


class DistMult(torch.nn.Module):
    """A class implementing the DistMult scoring function as a PyTorch module"""

    def __init__(self, n_rel, n_dim, device='cpu'):
        super(DistMult, self).__init__()
        self.device = device
        self.n_rel = n_rel
        self.n_dim = n_dim
        self.R = torch.nn.Parameter(
            torch.rand(n_rel, n_dim))  # random initialise trainable weights for scoring function
        self.to(self.device)

    def forward(self, embeddings, triples):
        """Take tensor of node embeddings and tensor of triples and return tensor or scores for each triple"""
        s, r, o = embeddings[triples[0, :], :], self.R[triples[1, :], :], embeddings[triples[2, :], :]
        return (s * r * o).sum(dim=-1)

    def extend(self, n_rel):
        r = DistMult(n_rel, self.n_dim)
        state_dict = r.state_dict()
        old_parameters = self.state_dict()['R'].detach() + 0.0
        state_dict['R'] = torch.cat([old_parameters, torch.randn(n_rel - self.n_rel, self.n_dim).to(self.device)])
        r.load_state_dict(state_dict)
        return r
