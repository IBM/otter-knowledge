import torch


class ESMProtein(torch.nn.Module):
    def __init__(self, repo_or_dir, model, repr_layer, device='cpu'):
        super(ESMProtein, self).__init__()
        self.device = device
        self.repo_or_dir = repo_or_dir
        self.model = model
        self.repr_layer = repr_layer
        self._model = None
        self._batch_converter = None
        self.to(self.device)

    def load_model(self):
        self._model, alphabet = torch.hub.load(self.repo_or_dir, self.model)
        self._batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        self._model.eval()
        self._model.forward_original = self._model.forward

    def get_embeddings(self, sequences):
        if not self._model:
            self.load_model()
        ids = ['ids_' + str(i) for i in range(len(sequences))]
        _, _, tensors = self._batch_converter(list(zip(ids, sequences)))
        results = self._model.forward_original(tensors, repr_layers=[self.repr_layer], return_contacts=True)
        token_representations = results["representations"][self.repr_layer]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, seq in enumerate(results['contacts']):
            seq_len = (tensors[i] == 2).nonzero()
            sequence_representations.append(token_representations[i, 1: seq_len].mean(0))

        return torch.stack(sequence_representations)
