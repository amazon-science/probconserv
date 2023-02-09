import math

import torch
from einops import rearrange
from torch import nn  # noqa: WPS458
from torch.distributions import Normal
from torch.nn import functional  # noqa: WPS458


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):  # noqa: WPS111
        return self.linear_layer(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, residual=False):
        super().__init__()
        layers = []
        in_sizes = [in_dim] + hidden_dims
        out_sizes = hidden_dims + [out_dim]
        for in_size, out_size in zip(in_sizes, out_sizes):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-1])
        self.residual = residual

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y += x
        return y


class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, dim_x, dim_y):
        super().__init__()
        input_dim = dim_x + dim_y
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init="relu")
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):  # noqa: WPS111
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)  # noqa: WPS111

        return mu, log_sigma, z


class DeterministicEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, dim_x, dim_y):
        super().__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(dim_x + dim_y, num_hidden)
        self.context_projection = Linear(dim_x, num_hidden)
        self.target_projection = Linear(dim_x, num_hidden)

    def forward(self, context_x, context_y, target_x):
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for self_att in self.self_attentions:
            encoder_input, _ = self_att(encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for cross_att in self.cross_attentions:
            query, _ = cross_att(keys, encoder_input, query)

        return query


class Decoder(nn.Module):
    def __init__(self, num_hidden, dim_x, dim_y):
        super().__init__()
        self.target_projection = Linear(dim_x, num_hidden)
        self.linears = nn.ModuleList(
            [
                Linear(num_hidden * 3, num_hidden * 3, w_init="relu") for _ in range(3)
            ]  # noqa: WPS221
        )
        self.final_projection = Linear(num_hidden * 3, dim_y * 2)
        self.dim_y = dim_y

    def forward(self, r, z, target_x, min_sigma=1e-3):  # noqa: WPS111
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = torch.cat((r, z, target_x), dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))

        # get mu and sigma
        y_pred = self.final_projection(hidden)
        y_mu, y_sigma = torch.split(y_pred, (self.dim_y, self.dim_y), -1)  # noqa: WPS221
        y_sigma = functional.softplus(y_sigma) + min_sigma

        return Normal(y_mu, y_sigma)


class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k):
        super().__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):  # noqa: WPS110
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        output = torch.bmm(attn, value)

        return output, attn


class Attention(nn.Module):
    def __init__(self, num_hidden, n_heads=4):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // n_heads
        self.n_heads = n_heads

        self.encoder_key = Linear(num_hidden, num_hidden, bias=False)
        self.encoder_value = Linear(num_hidden, num_hidden, bias=False)
        self.encoder_query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):  # noqa: WPS110
        pattern = "b n (nh nhpa) -> (nh b) n nhpa"
        key_enc = rearrange(self.encoder_key(key), pattern, nh=self.n_heads)
        value_enc = rearrange(self.encoder_value(value), pattern, nh=self.n_heads)
        query_enc = rearrange(self.encoder_query(query), pattern, nh=self.n_heads)

        # Get context vector
        output, attns = self.multihead(key_enc, value_enc, query_enc)

        # Concatenate all multihead context vector
        output = rearrange(output, "(nh b) sq nhpa -> b sq (nh nhpa)", nh=self.n_heads)
        # Concatenate context vector with input (most important)
        output = torch.cat([query, output], dim=-1)

        # Final linear
        output = self.final_linear(output)

        # Residual dropout & connection
        output = self.residual_dropout(output)
        output += query

        # Layer normalization
        output = self.layer_norm(output)

        return output, attns
