import torch


class BinaryClassifier(torch.nn.Module):
    """A class implementing the binary classifier scoring function as a PyTorch module"""

    def __init__(self, input_dim, device='cpu'):
        super(BinaryClassifier, self).__init__()
        self.device = device
        self.model = [torch.nn.Linear(2*input_dim, 1)]
        self.model = torch.nn.ModuleList(self.model)
        self.to(self.device)

    def forward(self, x):
        out = x
        for m in self.model:
            out = m(out)
        return out


class ClassifierHead(torch.nn.Module):
    """A class implementing the binary classifier scoring function as a PyTorch module"""

    def __init__(self, n_rel, n_dim, device='cpu'):
        super(ClassifierHead, self).__init__()
        self.device = device
        self.model = torch.nn.ModuleList([BinaryClassifier(n_dim, device=self.device) for i in range(n_rel)])
        self.n_rel = n_rel
        self.to(self.device)

    def forward(self, embeddings, triples):
        """Take tensor of node embeddings and tensor of triples and return tensor or scores for each triple"""
        s, r, o = embeddings[triples[0, :], :], triples[1, :], embeddings[triples[2, :], :]
        prediction = torch.zeros(len(s)).to(self.device)
        for i in range(self.n_rel):
            r_idx = r == i
            si = s[r_idx]
            oi = o[r_idx]
            if torch.sum(r_idx) > 0:
                x = torch.cat([si, oi], dim=-1)
                p = self.model[i](x)[:, 0]
                prediction[r_idx] = p
        return prediction

    def extend(self, n_rel):
        r = ClassifierHead(n_rel, self.n_dim, self.device)
        models = [m for m in self.model]
        for i in range(n_rel - self.n_rel):
            models.append(BinaryClassifier(self.n_dim, device=self.device))
        r.model = torch.nn.ModuleList(models)
        return r
