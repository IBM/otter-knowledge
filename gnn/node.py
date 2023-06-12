# classes that perform node embedding
import logging
from enum import Enum, unique
from typing import List, Optional, Union

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@unique
class ProjectionTypes(str, Enum):
    """ Projection type to project the embeddings into the graph.

    Possible values: projection, null
    """
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
    PROJECTION = "projection"
    NULL = "null"


class Node(torch.nn.Module):
    def __init__(self, modality: str, projection_type: Union[str, ProjectionTypes], embedding_size: int,
                 node_projection_embedding_size: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.embedding_type: Union[str, ProjectionTypes] = projection_type
            self.initial_embedding_size: int = embedding_size
        except ValueError:
            self.embedding_type: Union[str, ProjectionTypes] = ProjectionTypes.NULL
            self.initial_embedding_size: int = -1

        self.node_type: str = modality
        self.node_projection_embedding_size: int = node_projection_embedding_size
        if self.embedding_type == ProjectionTypes.PROJECTION:
            self.projection: torch.nn.Linear = torch.nn.Linear(self.initial_embedding_size,
                                                               self.node_projection_embedding_size)
        self.to(self.device)

    def forward(self, x: Union[List[Optional[torch.Tensor]], torch.Tensor]):
        # different types of node initial embedding can be implemented here
        if self.embedding_type == ProjectionTypes.PROJECTION:
            if isinstance(x, list):
                logging.warning(f"WARNING None in {self.embedding_type}, {x}, {self.node_type}, "
                                f"{self.initial_embedding_size}, {self.node_projection_embedding_size}")
                return self.projection(torch.zeros(len(x), self.initial_embedding_size).to(device))
            else:
                return self.projection(x)
        else:
            return torch.zeros(len(x), self.node_projection_embedding_size).to(device)
