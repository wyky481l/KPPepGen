import math
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from util.embedding.sequence import get_fasta_statis_without_pathogen, get_fasta_statis_with_pathogen


class SelfAttention(nn.Module):
    def __init__(self, n_head, dim_x, dim_hidden, dim_o, max_len):
        super().__init__()
        self.n_head = n_head
        self.dim_x = dim_x
        self.dim_q = dim_hidden
        self.dim_k = dim_hidden
        self.dim_v = dim_hidden
        self.dim_o = dim_o
        self.max_len = max_len

        self.weight_q = nn.Parameter(torch.Tensor(dim_x, self.dim_q))
        self.weight_k = nn.Parameter(torch.Tensor(dim_x, self.dim_k))
        self.weight_v = nn.Parameter(torch.Tensor(dim_x, self.dim_v))

        self.fc_q = nn.Linear(self.dim_q, n_head * self.dim_q)
        self.fc_k = nn.Linear(self.dim_k, n_head * self.dim_k)
        self.fc_v = nn.Linear(self.dim_v, n_head * self.dim_v)

        self.fc_o = nn.Linear(n_head * self.dim_v, dim_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        q = torch.matmul(x, self.weight_q)
        k = torch.matmul(x, self.weight_k)
        v = torch.matmul(x, self.weight_v)

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        # [batch, len, dim * n_head]

        q = q.view(batch_size, self.max_len, self.n_head, self.dim_q).permute(2, 0, 1, 3).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            self.dim_q)
        k = k.view(batch_size, self.max_len, self.n_head, self.dim_k).permute(2, 0, 1, 3).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            self.dim_k)
        v = v.view(batch_size, self.max_len, self.n_head, self.dim_v).permute(2, 0, 1, 3).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            self.dim_v)
        # [batch * n_head, len, dim]

        attention = torch.bmm(q, k.transpose(1, 2))
        # [batch * n_head, len, len]

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
            # [batch, len, len] -> [batch * n_head, len, len]
            attention = attention.masked_fill(mask, -1e20)
            # -float('inf')

        attn = F.softmax(attention, dim=-1)
        attn = torch.where(mask, attn, torch.zeros_like(attn, device=x.device))

        output = torch.bmm(attn, v)

        output = output.view(self.n_head, batch_size, self.max_len, self.dim_v).permute(1, 2, 0, 3).contiguous().view(
            batch_size, self.max_len, -1)
        output = self.fc_o(output)

        return output, attn


class SinusoidalEmb(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, embed_size, 2).float() / embed_size)
        )

    def forward(self, time_step, batch=None, edge_index=None):
        self.inv_freq = self.inv_freq.to(time_step.device)

        if batch is not None:
            if edge_index is not None:
                edge_graph = batch.index_select(0, edge_index[0])
                edge_time_step = time_step.index_select(0, edge_graph)
                time_step = edge_time_step.unsqueeze(dim=-1)
            else:
                node_time_step = time_step.index_select(0, batch)
                time_step = node_time_step.unsqueeze(dim=-1)
        else:
            time_step = time_step.unsqueeze(dim=-1)

        pos_enc_a = torch.sin(time_step.repeat(1, self.embed_size // 2) * self.inv_freq)
        pos_enc_b = torch.cos(time_step.repeat(1, self.embed_size // 2) * self.inv_freq)
        pos_enc = torch.cat((pos_enc_a, pos_enc_b), dim=-1)

        return pos_enc


class SinPosEmb(nn.Module):
    def __init__(self, max_length, dim):
        super().__init__()
        self.max_length = max_length
        self.dim = dim
        self.positional_emb = None

    def forward(self, device=None):
        if self.positional_emb is None:
            self.positional_emb = self.positional_encoding(self.max_length, device)

        return self.positional_emb

    def positional_encoding(self, seq_len, device):
        d_model = self.dim
        assert seq_len <= self.max_length, print("input sequence length over 50 ")

        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)

        pe = torch.zeros(seq_len, d_model, device=device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


def get_sentence_emb(seq_emb, seq_len_list, seq_attn=None):
    seq_size = len(seq_emb)
    sentence_emb_list = []

    if seq_attn is None:
        for index in range(seq_size):
            seq_len = seq_len_list[index]
            emb = seq_emb[index, :seq_len, :].mean(dim=0)
            emb = emb.unsqueeze(0)
            sentence_emb_list.append(emb)

        output_emb = torch.cat(sentence_emb_list, dim=0)

    else:
        for index in range(seq_size):
            seq_len = seq_len_list[index]
            attn = seq_attn[index].mean(dim=0)[:seq_len]
            emb = seq_emb[index, :seq_len, :]
            attn_emb = attn @ emb

            attn_emb = attn_emb.unsqueeze(0)
            sentence_emb_list.append(attn_emb)

        output_emb = torch.cat(sentence_emb_list, dim=0)

    return output_emb


class SeqFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()
        self.dim_list = [input_dim, input_dim // 2, input_dim // 4, output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class ConditionFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, input_dim * 4, input_dim * 2, input_dim, output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class AdapterFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 2, output_dim * 2, output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class SeqNoise(nn.Module):
    def __init__(self, n_class=20):
        super().__init__()
        # dud: discrete uniform distribution
        # ddd: discrete data-based distribution
        self.n_class = n_class

        self.pathogen_freq_dict = get_fasta_statis_with_pathogen()
        seq_length_freq, self.aa_count_freq = get_fasta_statis_without_pathogen()

    def get_noise(self, device=None, pathogen_type=None, uniform_state=False):

        if uniform_state:
            noise = torch.ones([self.n_class], device=device) / self.n_class
            return noise.unsqueeze(dim=0)

        if pathogen_type is not None:
            noise = torch.tensor(self.pathogen_freq_dict[pathogen_type], device=device).unsqueeze(dim=0)
        else:
            noise = torch.tensor(self.aa_count_freq, device=device).unsqueeze(dim=0)

        return noise
