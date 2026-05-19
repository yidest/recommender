"""Neural Collaborative Filtering model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NeuralCollaborativeFiltering(nn.Module):
    """User/movie embeddings followed by an MLP scoring network."""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.scoring_network = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, mean=0.0, std=0.01)

    def forward(self, user_indices: torch.Tensor, movie_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_indices)
        movie_vectors = self.movie_embedding(movie_indices)
        features = torch.cat([user_vectors, movie_vectors], dim=1)
        return self.scoring_network(features).squeeze(-1)


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -F.logsigmoid(positive_scores - negative_scores).mean()
