import torch
from einops import reduce
import numpy as np


class SoftmaxAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super(SoftmaxAttention, self).__init__()

    def forward(self, q, k, v):
        scores = torch.einsum('b h i d, b h j d-> b h i j', q, k) / q.size(-1) ** 0.5
        weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('b h i j, b h j d-> b h i d', weights, v)


class LinearAttention(torch.nn.Module):
    def __init__(self, eps=1e-6, **kwargs):
        super(LinearAttention, self).__init__()
        self.eps = eps

    def forward(self, q, k, v):
        q = torch.nn.functional.elu(q) + 1 + self.eps
        k = torch.nn.functional.elu(k) + 1 + self.eps

        v_length = v.size(-1)
        v = v / v_length
        kv = torch.einsum('b h n i, b h n j -> b h i j', k, v)
        normalizer = 1 / torch.einsum('b h n d, b h d-> b h n', q, k.sum(dim=2))
        return torch.einsum('b h n d, b h d e, b h n -> b h n e', q, kv, normalizer) * v_length


class LinformerAttention(SoftmaxAttention):
    def __init__(self, seq_length, p_dim, **kwargs):
        super(LinformerAttention, self).__init__()
        self.seq_projection = torch.nn.parameter.Parameter(torch.randn(seq_length, p_dim))
        torch.nn.init.kaiming_normal_(self.seq_projection, mode='fan_out')

    def forward(self, q, k, v):
        k = torch.einsum('bhnd,nk->bhkd', k, self.seq_projection)
        v = torch.einsum('bhnd,nk->bhkd', v, self.seq_projection)
        return super().forward(q, k, v)


class RandomFeatureAttention(torch.nn.Module):
    def __init__(self, heads, head_dim, **kwargs):
        super(RandomFeatureAttention, self).__init__()
        self.projections = torch.nn.parameter.Parameter(torch.randn(size=(heads, head_dim, head_dim)), requires_grad=False)

    def forward(self, q, k, v):
        q = torch.einsum('b h n d, h d e -> b h n e', q, self.projections)
        k = torch.einsum('b h n d, h d e -> b h n e', k, self.projections)
        phi_q = torch.cat([torch.sin(q), torch.cos(q)], dim=-1) / q.size(-1) ** 0.5
        phi_k = torch.cat([torch.sin(k), torch.cos(k)], dim=-1) / q.size(-1) ** 0.5
        kv = torch.einsum('b h n i, b h n j -> b h i j', phi_k, v)
        return torch.einsum('b h n e, b h e d -> b h n d', phi_q, kv)


class NystromAttention(torch.nn.Module):
    def __init__(self, seq_length, p_dim, **kwargs):
        super(NystromAttention, self).__init__()
        self.seq_length = seq_length
        self.p_dim = p_dim

    def forward(self, q, k, v):
        q = q / q.size(-1) ** 0.5

        l = np.ceil(self.seq_length / self.p_dim).astype(int)

        remainder = self.seq_length % self.p_dim
        if remainder > 0:
            padding = self.p_dim - remainder
            q_landmarks = reduce(torch.nn.functional.pad(q, (0, 0, padding, 0), v=0),
                                 'b h (n l) d -> b h n d', 'mean', l=l)
            k_landmarks = reduce(torch.nn.functional.pad(k, (0, 0, padding, 0), v=0),
                                 'b h (n l) d -> b h n d', 'mean', l=l)
        else:
            q_landmarks = reduce(q, 'b h (n l) d -> b h n d', 'mean', l=l)
            k_landmarks = reduce(k, 'b h (n l) d -> b h n d', 'mean', l=l)

        F = torch.einsum('b h i d, b h j d -> b h i j', q, k_landmarks).softmax(dim=-1)
        A = torch.linalg.pinv(torch.einsum('b h i d, b h j d -> b h i j', q_landmarks, k_landmarks).softmax(dim=-1))
        B = torch.einsum('b h i d, b h j d -> b h i j', q_landmarks, k).softmax(dim=-1)
        return (F @ A) @ (B @ v)
