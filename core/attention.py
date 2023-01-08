import torch


def full_attention(query, key, value):
    num_heads = query.size(1)
    scores = torch.einsum('bhid,bhjd->bhij', query, key) / num_heads ** 0.5
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhij,bhjd->bhid', weights, value)


def linear_full_attention(query, key, value, eps=1e-6):
    query = torch.nn.functional.elu(query) + 1
    key = torch.nn.functional.elu(key) + 1

    v_length = value.size(-1)
    value = value / v_length
    kv = torch.einsum('bhni,bhnj->bhij', key, value)
    normalizer = 1 / torch.einsum('bhnd,bhd->bhn', query, key.sum(dim=2))
    queried_values = torch.einsum('bhnd,bhde,bhn->bhne', query, kv, normalizer) * v_length
    return queried_values
