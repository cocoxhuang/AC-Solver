import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.distributions import Categorical

class FFN(nn.Module):
    def __init__(self, input_size: int, nodes: list, output_size: int):
        '''
        Args:
            input_size: size of the input layer
            nodes: list of integers representing the number of nodes in each hidden layer
            output_size: size of the output layer
        '''
        super(FFN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, nodes[0]))
        assert len(nodes) >= 1, "At least one hidden layer is required"
        for i in range(1, len(nodes)):
            self.layers.append(nn.Linear(nodes[i-1], nodes[i]))
        self.layers.append(nn.Linear(nodes[-1], output_size))

    def forward(self, x):
        x = x.float()
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # Use the final linear layer to produce logits
        logits = self.layers[-1](x)
        return logits
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ResNet model is not implemented yet.")

    def forward(self, x):
        # Placeholder for forward pass
        raise NotImplementedError("ResNet model is not implemented yet.")

# ------------Transformer Related Models ------------
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, positional_encoding="sinusoidal"):
        super().__init__()
        self.positional_encoding = positional_encoding
        if positional_encoding == "sinusoidal":
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
            pe = torch.zeros(max_len, embedding_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif positional_encoding == "absolute_learned":
            self.pe = nn.Embedding(max_len, embedding_dim)
            self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        if self.positional_encoding == "sinusoidal":
            return x + self.pe[:x.size(1)]
        elif self.positional_encoding == "absolute_learned":
            position_ids = self.position_ids[:x.size(1)]
            return x + self.pe(position_ids).unsqueeze(0)

# ------------Dual Ring Models (Xatt from both r1 and r2) ------------
class AbsoluteSelfAttention(nn.Module):
    def __init__(self, num_heads, head_dim, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.qkv_linear = nn.Linear(num_heads * head_dim, 3 * num_heads * head_dim)
        self.out_linear = nn.Linear(num_heads * head_dim, num_heads * head_dim)

    def forward(self, x, mask):
        B, L, D = x.shape
        H = self.num_heads
        D_head = self.head_dim
        
        # turn off positional encoding at every attention
        # pe = fixed_positional_encoding(L, D, device=x.device)
        # x = x + pe
        
        qkv = self.qkv_linear(x).view(B, L, 3, H, D_head)
        q, k, v = qkv.split(1, dim=2)
        q = q.squeeze(2).transpose(1, 2)  # [B, H, L, D_head]
        k = k.squeeze(2).transpose(1, 2)
        v = v.squeeze(2).transpose(1, 2)
        
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(~mask_expanded, -1e9)
        
        attn = F.softmax(scores / math.sqrt(D_head), dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, L, H * D_head)
        
        return self.out_linear(out)

class AbsoluteCrossAttention(nn.Module):
    def __init__(self, num_heads, head_dim, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.q_linear = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.kv_linear = nn.Linear(num_heads * head_dim, 2 * num_heads * head_dim)
        self.out_linear = nn.Linear(num_heads * head_dim, num_heads * head_dim)

    def forward(self, q_seq, kv_seq, q_mask=None, kv_mask=None):
        B, Lq, D = q_seq.shape
        Lv = kv_seq.shape[1]
        H = self.num_heads
        D_head = self.head_dim
        
        # turn off positional encoding at every attention 
        # q_pe = fixed_positional_encoding(Lq, D, device=q_seq.device)
        # kv_pe = fixed_positional_encoding(Lv, D, device=kv_seq.device)
        # q_seq = q_seq + q_pe
        # kv_seq = kv_seq + kv_pe
        
        q = self.q_linear(q_seq).view(B, Lq, H, D_head).transpose(1, 2)
        kv = self.kv_linear(kv_seq).view(B, Lv, 2, H, D_head)
        k, v = kv.split(1, dim=2)
        k = k.squeeze(2).transpose(1, 2)
        v = v.squeeze(2).transpose(1, 2)
        
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        
        if kv_mask is not None:
            kv_mask_expanded = kv_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Lv]
            scores = scores.masked_fill(~kv_mask_expanded, -1e9)
        if q_mask is not None:
            q_mask_expanded = q_mask.unsqueeze(1).unsqueeze(3)  # [B, 1, Lq, 1]
            scores = scores.masked_fill(~q_mask_expanded, -1e9)
        
        attn = F.softmax(scores / math.sqrt(D_head), dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, Lq, H * D_head)
        
        return self.out_linear(out)
        
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, out_dim, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, out_dim)
        if activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "tanh":
            self.act_fn = torch.tanh
        else:
            raise NotImplementedError(f"{activation} activation not implemented.")

    def forward(self, x):
        return self.linear2(self.act_fn(self.linear1(x)))
    
class DualRingBlock(nn.Module):
    def __init__(self, num_heads, head_dim, mlp_dim, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.max_len = max_len
        embedding_dim = num_heads * head_dim
        
        # change: do not normalize inputs first
        # self.norm1 = nn.LayerNorm(embedding_dim)
        # self.norm2 = nn.LayerNorm(embedding_dim)
        # Layer norms
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.norm5 = nn.LayerNorm(embedding_dim)
        self.norm6 = nn.LayerNorm(embedding_dim)
        
        # Attention layers
        self.self_attn1 = AbsoluteSelfAttention(num_heads, head_dim, max_len)
        self.self_attn2 = AbsoluteSelfAttention(num_heads, head_dim, max_len)
        self.cross_attn1 = AbsoluteCrossAttention(num_heads, head_dim, max_len)
        self.cross_attn2 = AbsoluteCrossAttention(num_heads, head_dim, max_len)
        
        # MLP layers
        self.mlp1 = MLP(embedding_dim, mlp_dim, embedding_dim)
        self.mlp2 = MLP(embedding_dim, mlp_dim, embedding_dim)

    def forward(self, x1, x2, mask1, mask2):
        
        # Change: Do not normalize inputs first
        # x1 = self.norm1(x1)
        # x2 = self.norm2(x2)
        
        # Self-attention
        x1 = self.norm3(x1 + self.self_attn1(x1, mask1))
        x2 = self.norm4(x2 + self.self_attn2(x2, mask2))
        
        # Cross-attention
        x1 = self.norm5(x1 + self.cross_attn1(x1, x2, q_mask=mask1, kv_mask=mask2))
        x2 = self.norm6(x2 + self.cross_attn2(x2, x1, q_mask=mask2, kv_mask=mask1))

        x1 = x1 + self.mlp1(x1)
        x2 = x2 + self.mlp2(x2)

        return x1, x2

def get_cyclic_neighbor(arr, mask, shift):
    """Get cyclic neighbors with proper masking."""
    device = arr.device
    B, L = arr.shape
    lengths = mask.sum(dim=1)  # [B]
    
    base_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    shifted_idx = (base_idx + shift) % lengths.unsqueeze(1)
    final_idx = torch.where(mask, shifted_idx, base_idx)
    
    return torch.gather(arr, 1, final_idx)

class DualRingActor(nn.Module):
    def __init__(self, activation: str = "gelu", num_layers: int = 2, 
                 num_heads : int = 4, head_dim : int = 8, 
                 mlp_dim : int =32, vocab_size : int = 5, max_len : int = 32, 
                 n_actions : int = None):
        super().__init__()
        self.activation = activation    
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.embedding_dim = num_heads * head_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_actions = n_actions if n_actions is not None else (max_len // 2) * (max_len // 2) * 2 * 2  # total action space size
        
        # Embedding and (Change) positional encoding
        self.embed = nn.Embedding(vocab_size, self.embedding_dim)
        self.pe = PositionalEncoding(self.embedding_dim, max_len, positional_encoding="none")
        
        # DualRing blocks
        self.dual_ring_blocks = nn.ModuleList([
            DualRingBlock(num_heads, head_dim, mlp_dim, max_len)
            for _ in range(num_layers)
        ])
        
        # Actor head layers
        # Change: hidden layer size equal to embedding_dim
        self.out = MLP(self.embedding_dim * 2, self.embedding_dim, self.n_actions)   
        
        # Critic head layers 
        # Not training critic head for now but keeping the code
        # self.critic_dense1 = nn.Linear(2 * self.embedding_dim, 256)
        # self.critic_dense2 = nn.Linear(256, 256)
        # self.critic_dense3 = nn.Linear(256, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.out:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def logits_mask(self,r1_raw, r2_raw, mask1, mask2):
        B, L_half = r1_raw.shape

        r1_exp = r1_raw.unsqueeze(2)  # [B, L_half, 1]
        r2_exp = r2_raw.unsqueeze(1)  # [B, 1, L_half]
        r1_broadcast = r1_exp.expand(B, L_half, L_half)
        r2_broadcast = r2_exp.expand(B, L_half, L_half)

        # Get cyclic neighbors for cancellation check
        r1_next = get_cyclic_neighbor(r1_raw, mask1, shift=1)  # [B, L_half]
        r2_prev = get_cyclic_neighbor(r2_raw, mask2, shift=-1)  # [B, L_half]
        r1_next_broadcast = r1_next.unsqueeze(2).expand(B, L_half, L_half)
        r2_prev_broadcast = r2_prev.unsqueeze(1).expand(B, L_half, L_half)

        # Check for cancellation conditions
        no_cancellation_mask_j0 = (r1_next_broadcast != -r2_prev_broadcast)
        no_cancellation_mask_j1 = (r1_next_broadcast != r2_prev_broadcast)

        mask_j0 = (r1_broadcast == -r2_broadcast) & no_cancellation_mask_j0
        mask_j1 = (r1_broadcast == r2_broadcast) & no_cancellation_mask_j1

        semantic_mask = torch.stack([mask_j0, mask_j1], dim=-1)  # [B, L_half, L_half, 2]

        # Padding masks for i and j positions
        mask1_b = mask1.unsqueeze(2)  # [B, L_half, 1]
        mask2_b = mask2.unsqueeze(1)  # [B, 1, L_half]
        padding_mask = mask1_b & mask2_b  # [B, L_half, L_half]

        # Combine semantic mask with padding mask to get valid (i,j) actions per type
        base_mask = semantic_mask & padding_mask.unsqueeze(-1)  # [B, L_half, L_half, 2]
        logits_mask = base_mask.unsqueeze(3).expand(B, L_half, L_half, 2, 2)
        logits_mask = logits_mask.reshape(B, -1)  # [B, total_action_dim]

        return logits_mask
    
    def masked_mean(self, x, mask, padding_denom=True):
        '''Compute masked mean over the sequence length dimension.'''
        mask_float = mask.float()
        pad_denom = 1e-6 if padding_denom else 0.0
        return (x * mask_float.unsqueeze(-1)).sum(dim=1) / (mask_float.sum(dim=1, keepdim=True) + pad_denom)
    
    def forward(self, input_seq):
        B, L = input_seq.shape
        L_half = L // 2
        device = input_seq.device

        # Split rings and build masks
        r1_raw, r2_raw = input_seq[:, :L_half], input_seq[:, L_half:]
        mask1 = (r1_raw != 0)  # [B, L_half]
        mask2 = (r2_raw != 0)  # [B, L_half]
        
        x1 = self.embed((r1_raw).long())  # [B, L_half, D]
        x2 = self.embed((r2_raw).long())  # [B, L_half, D]

        for block in self.dual_ring_blocks:
            x1, x2 = block(x1, x2, mask1, mask2)    # [B, L_half, D], [B, L_half, D]

        # Change: now just do a pooling instead of (masked) full action space logits
        # TODO: recover the full action space logits

        pooled1 = self.masked_mean(x1, mask1) # [B, D]
        pooled2 = self.masked_mean(x2, mask2) # [B, D]
        x = torch.cat([pooled1, pooled2], dim=-1) # [B, 2D]

        logits = self.out(x)  # [B, n_actions]
        # pi = Categorical(logits=logits)

        # x1 = x1.unsqueeze(2).expand(B, L_half, L_half, x1.shape[-1])  # [B, L_half, L_half, D]
        # x2 = x2.unsqueeze(1).expand(B, L_half, L_half, x2.shape[-1])  # [B, L_half, L_half, D]
        # x = torch.cat([x1, x2], dim=-1)  # [B, L_half, L_half, 2D]
        # x = self.out(x)  # [B, L_half, L_half, 4]
        # logits = x.reshape(B, -1)  # [B, total_action_dim]
        # logits_mask = self.logits_mask(r1_raw, r2_raw, mask1, mask2)  # [B, total_action_dim] 
        # logits = torch.where(logits_mask, logits, torch.tensor(-1e9, device=device))
        
        # not training critic head for now but keeping the code
        # pooled1 = masked_mean(x1, mask1) # [B, D]
        # pooled2 = masked_mean(x2, mask2) # [B, D]
        # x = torch.cat([pooled1, pooled2], dim=-1) # [B, 2D]

        # critic = self.critic_dense1(x)
        # critic = act_fn(critic)
        # critic = self.critic_dense2(critic)
        # critic = act_fn(critic)
        # critic = self.critic_dense3(critic).squeeze(-1)

        # return pi 
        return logits
        # return pi, critic
    
    def generate(self, input_seq, method: str = 'greedy'):
        logits = self.forward(input_seq)
        pi = Categorical(logits=logits)
        if method == 'greedy':
            action = torch.argmax(pi.logits, dim=1)
        elif method == 'sample':
            action = pi.sample()
        return action

if __name__ == "__main__":
    model = DualRingActor(n_actions=336)
    sample_input = torch.randint(0, 5, (2, 32))
    print(sample_input)
    action = model.generate(sample_input)
    print(action)