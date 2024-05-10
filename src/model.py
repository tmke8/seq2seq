import math

import torch
from torch import Tensor, nn

__all__ = ["Translator", "create_mask"]


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class Translator(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embed_size: int,
        num_heads: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        decoder_norm = nn.LayerNorm(embed_size)
        custom_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            custom_decoder=custom_decoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.ff = nn.Linear(embed_size, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.pos_enc(self.src_embedding(src))
        tgt_emb = self.pos_enc(self.tgt_embedding(trg))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        return self.ff(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        embed = self.src_embedding(src)

        pos_enc = self.pos_enc(embed)

        return self.transformer.encoder(pos_enc, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        embed = self.tgt_embedding(tgt)

        pos_enc = self.pos_enc(embed)

        return self.transformer.decoder(pos_enc, memory, tgt_mask)


def create_mask(
    src: Tensor, tgt: Tensor, pad_idx: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create masks for input into model"""

    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate the mask
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
