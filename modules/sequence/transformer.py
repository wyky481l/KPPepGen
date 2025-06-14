from torch import nn
from torch.nn import Softplus
from modules.sequence.encoder import SinusoidalEmb, SinPosEmb


class SequenceFFN(nn.Module):
    def __init__(self, n_emb, dropout):
        super().__init__()
        self.L1 = nn.Linear(n_emb, 4 * n_emb)
        self.L2 = nn.Linear(4 * n_emb, 2 * n_emb)
        self.L3 = nn.Linear(2 * n_emb, n_emb)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu((self.L1(x)))
        x = self.relu((self.L2(x)))
        x = self.dropout(self.L3(x))
        return x


class SequenceBlock(nn.Module):
    def __init__(
            self,
            n_emb,
            n_head,
            ffn_drop,
            block_drop,
    ):
        super().__init__()

        self.dropout = nn.Dropout(block_drop)

        self.ln1 = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_emb, elementwise_affine=False)

        self.ffn = SequenceFFN(n_emb=n_emb, dropout=ffn_drop)
        self.act = Softplus()

        self.linear_t = nn.Linear(n_emb, n_emb)
        self.linear_pos = nn.Linear(n_emb, n_emb)

        self.attn = nn.MultiheadAttention(embed_dim=n_emb, num_heads=n_head)

    def forward(self, x, time_step, padding_mask, emb_t, emb_pos):
        pos_emb = emb_pos(x.device)
        x = x + pos_emb

        if time_step is not None:
            time_emb = emb_t(time_step)
            time_emb = time_emb.unsqueeze(1).repeat(1, 50, 1)
            x = x + time_emb

        x = x.permute(1, 0, 2)

        a, att = self.attn(query=x, key=x, value=x, key_padding_mask=padding_mask)
        x = self.dropout(x + a).permute(1, 0, 2)
        x = self.ln1(x)

        x = self.dropout(x + self.ffn(x))
        x = self.ln2(x)

        return x, None


class SeqTransformer(nn.Module):
    def __init__(
            self,
            output_dim=512,
            n_emb=512,
            n_head=8,
            ffn_drop=0.1,
            block_drop=0.1,
            diff_status=False,
            n_block=8,
            emb_type="pos_emb",
            n_seq_max=50,

    ):
        super().__init__()

        self.n_block = n_block

        self.output_emb = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, output_dim),
        )

        self.blocks = nn.Sequential(*[SequenceBlock(
            n_emb=n_emb,
            n_head=n_head,
            ffn_drop=ffn_drop,
            block_drop=block_drop,
        ) for _ in range(n_block)])

        if diff_status:
            self.emb_t = SinusoidalEmb(n_emb)
        else:
            self.emb_t = None

        if emb_type == "pos_emb":
            self.emb_pos = SinPosEmb(n_seq_max, n_emb)
        else:
            self.emb_pos = nn.Embedding(n_seq_max, n_emb)

    def forward(self, x_emb, time_step, padding_mask):
        for index in range(self.n_block):
            x_emb, attn_weight = self.blocks[index](x_emb, time_step, padding_mask, self.emb_t,
                                                    self.emb_pos)

        output = self.output_emb(x_emb)
        return output
