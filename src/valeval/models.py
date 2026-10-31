"""Partie C : réseaux de neurones sur embeddings de mots."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_VALEURS = 20


def couche_embedding(n_mots: int, dim: int, vecteurs=None, gele: bool = False) -> nn.Embedding:
    """Embeddings aléatoires, ou pré-entraînés si vecteurs est fourni."""
    if vecteurs is None:
        return nn.Embedding(n_mots, dim, padding_idx=0)
    return nn.Embedding.from_pretrained(torch.as_tensor(vecteurs), freeze=gele, padding_idx=0)


class Etats(nn.Module):
    """Remplace l'embedding quand l'entrée est déjà une suite de vecteurs (Partie D2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.embedding_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BiLSTM(nn.Module):
    """C1 : lit le texte dans les deux sens, puis résume la séquence en un vecteur."""

    def __init__(self, embedding: nn.Embedding, cache: int = 128, pooling: str = "max", dropout: float = 0.3):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, cache, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(2 * cache, N_VALEURS)

    def encoder(self, ids: torch.Tensor, masque: torch.Tensor) -> tuple:
        x = self.dropout(self.embedding(ids))
        longueurs = masque.sum(1).cpu()
        paquet = pack_padded_sequence(x, longueurs, batch_first=True, enforce_sorted=False)
        etats, (dernier, _) = self.lstm(paquet)
        etats, _ = pad_packed_sequence(etats, batch_first=True, total_length=ids.size(1))
        return etats, dernier

    def forward(self, ids: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        etats, dernier = self.encoder(ids, masque)
        if self.pooling == "dernier":
            resume = torch.cat([dernier[0], dernier[1]], dim=1)   # fin de la lecture dans chaque sens
        else:
            # pour chaque dimension, la plus forte activation sur toute la séquence
            resume = etats.masked_fill(~masque.unsqueeze(-1), -1e4).max(dim=1).values
        return self.sortie(self.dropout(resume))


class Attention(nn.Module):
    """Pooling attentionnel : alpha_i = softmax(v' tanh(W h_i)), c = somme des alpha_i h_i."""

    def __init__(self, dim: int, dim_attention: int = 128):
        super().__init__()
        self.W = nn.Linear(dim, dim_attention)
        self.v = nn.Linear(dim_attention, 1, bias=False)

    def forward(self, etats: torch.Tensor, masque: torch.Tensor) -> tuple:
        scores = self.v(torch.tanh(self.W(etats))).squeeze(-1)    # un score par mot
        scores = scores.masked_fill(~masque, float("-inf"))        # jamais d'attention sur le padding
        alpha = scores.softmax(dim=1)                              # poids positifs, de somme 1
        return (alpha.unsqueeze(-1) * etats).sum(dim=1), alpha


class BiLSTMAttention(BiLSTM):
    """C2 : comme C1, mais le résumé est une moyenne des mots pondérée par l'attention."""

    def __init__(self, embedding: nn.Embedding, cache: int = 128, dropout: float = 0.3):
        super().__init__(embedding, cache, "attention", dropout)
        self.attention = Attention(2 * cache)

    def forward(self, ids: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        etats, _ = self.encoder(ids, masque)
        return self.sortie(self.dropout(self.attention(etats, masque)[0]))

    def poids(self, ids: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        return self.attention(self.encoder(ids, masque)[0], masque)[1]


class EncodagePositionnel(nn.Module):
    """Ajoute à chaque position un motif sinusoïdal fixe, pour que le modèle connaisse l'ordre des mots."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        frequence = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * frequence)
        pe[:, 1::2] = torch.cos(position * frequence)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class MiniTransformer(nn.Module):
    """C3 : 2 couches, d_model 128, 4 têtes ; positionnel=False pour l'ablation."""

    def __init__(self, embedding: nn.Embedding, d_model: int = 128, tetes: int = 4, couches: int = 2,
                 pooling: str = "moyenne", positionnel: bool = True, dropout: float = 0.3):
        super().__init__()
        self.embedding = embedding
        self.projection = nn.Linear(embedding.embedding_dim, d_model)
        self.positions = EncodagePositionnel(d_model) if positionnel else nn.Identity()
        couche = nn.TransformerEncoderLayer(d_model, tetes, dim_feedforward=2 * d_model, dropout=dropout,
                                            batch_first=True)
        self.encodeur = nn.TransformerEncoder(couche, couches, enable_nested_tensor=False)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(d_model, N_VALEURS)

    def forward(self, ids: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        x = self.positions(self.projection(self.dropout(self.embedding(ids))))
        etats = self.encodeur(x, src_key_padding_mask=~masque)
        if self.pooling == "max":
            resume = etats.masked_fill(~masque.unsqueeze(-1), -1e4).max(dim=1).values
        else:
            resume = (etats * masque.unsqueeze(-1)).sum(1) / masque.sum(1, keepdim=True)
        return self.sortie(self.dropout(resume))


class CharCNN(nn.Module):
    """C4 : filtres glissant sur les caractères, de 3, 5 et 7 lettres."""

    def __init__(self, n_chars: int, dim: int = 32, filtres: int = 128, tailles=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(n_chars, dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim, filtres, t, padding=t // 2) for t in tailles)
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(filtres * len(tailles), N_VALEURS)

    def forward(self, ids: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(ids)).transpose(1, 2)        # (lot, dim, longueur)
        # pour chaque filtre, sa plus forte réponse sur tout le texte
        resumes = [torch.relu(c(x)).masked_fill(~masque.unsqueeze(1), 0).max(dim=2).values for c in self.convs]
        return self.sortie(self.dropout(torch.cat(resumes, dim=1)))
