from einops import rearrange

from .attention import *


class MultiHeadAttention(torch.nn.Module):
    attention = {
        'softmax': SoftmaxAttention,
        'linear': LinearAttention,
        'linformer': LinformerAttention,
        'random_feature': RandomFeatureAttention,
        'nystrom': NystromAttention
    }

    def __init__(self, seq_length, dim, heads, attention_type):
        super(MultiHeadAttention, self).__init__()
        self.attention_type = attention_type
        assert dim % heads == 0
        head_dim = dim // heads
        self.heads = heads

        self.qkv_projection = torch.nn.Linear(dim, dim * 3)
        self.out_projection = torch.nn.Linear(dim, dim)

        self.attention_layer = self.attention[attention_type](seq_length=seq_length, p_dim=seq_length//8, heads=heads, head_dim=head_dim)

    def forward(self, x):
        qkv = self.qkv_projection(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        out = self.attention_layer(q, k, v)
        return self.out_projection(rearrange(out, 'b h n d -> b n (h d)'))


class TransformerBlock(torch.nn.Module):
    def __init__(self, seq_length, dim, heads, feedforward_dim, attention_type):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(seq_length, dim, heads, attention_type)

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(dim, feedforward_dim),
            torch.nn.GELU(),
            torch.nn.Linear(feedforward_dim, dim),
        )

        self.mha_norm = torch.nn.LayerNorm(dim)
        self.feedworward_norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.mha(self.mha_norm(x))
        x = x + self.feedforward(self.feedworward_norm(x))
        return x


class ViT(torch.nn.Module):
    def __init__(self, image_size, num_classes, dim, depth, heads, feedforward_dim, attention_type='full', channels=3):
        super(ViT, self).__init__()

        self.embedding_projection = torch.nn.Linear(3, dim)

        self.transformer = torch.nn.Sequential()
        for i in range(depth):
            self.transformer.add_module(f'block_{i}', TransformerBlock(image_size ** 2, dim, heads, feedforward_dim, attention_type))

        self.output_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, num_classes)
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, image_size ** 2, dim))

    def forward(self, img):
        x = img.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)
        x = self.embedding_projection(x)
        x = x + self.pos_embedding

        x = self.transformer(x)

        return self.output_projection(x.mean(dim=1))
