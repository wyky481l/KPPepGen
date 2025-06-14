from torch import nn
import torch.nn.functional as F
import torch


class SeqTermGOProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class StructTermGOProjector(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class GoTermGOProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class SeqTermPathogenProjector(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class StructTermPathogenProjector(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class PathogenTermProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=128, activation="gelu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim * 4, output_dim * 2, output_dim, output_dim]

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


class RelationGOProjector(nn.Module):
    def __init__(self, input_dim=6, output_dim=128, activation="silu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, output_dim // 4, output_dim // 2, output_dim, output_dim]

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


def transe_loss(head, relation, tail, neg_tail, max_score=10):
    if relation is None:
        relation = torch.zeros_like(head)

    pos_score = (head + relation - tail).norm(p=1, dim=-1)
    pos_loss = -1.0 * F.logsigmoid(max_score - pos_score)

    neg_score = (head + relation - neg_tail).norm(p=1, dim=-1)
    neg_loss = -1.0 * F.logsigmoid(neg_score - max_score)

    total_loss = pos_loss.mean() + neg_loss.mean()

    return total_loss, pos_loss, neg_loss
