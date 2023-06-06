from logging import getLogger

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class MorganFingerprint:
    def __init__(self, shape=2048, radius=2):
        self.shape = shape
        self.radius = radius
        self.logger = getLogger(__name__)

    @staticmethod
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return smiles

    def morgan_finger_print(self, smile) -> torch.Tensor:
        try:
            smile = self.canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            self.logger.warning(f"rdkit not found this smiles for morgan: {smile} convert to all 0 features")
            features = np.zeros((self.shape,))
        return torch.tensor(features, dtype=torch.float32)

    def get_embeddings(self, smiles):
        return [self.morgan_finger_print(s) for s in smiles]
