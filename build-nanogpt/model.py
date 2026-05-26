import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
import __main__

class RotaryEmbedding(nn.Module):
    """rotary positional embedding (rope).

    precomputes cos/sin tables of shape (1, 1, max_pos, head_dim) so they
    broadcast cleanly over (b, n_head, t, head_dim) inside attention.
    """

    def __init__(self, dim, base=10000.0, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        # standard rope inverse-frequency schedule over even dims
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_pos, dim/2)
        # duplicate the freqs across the head dim so we can rotate halves
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_pos, dim)
        # buffers, not parameters: not trained, not saved by default
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    @staticmethod
    def _rotate_half(x):
        # rotate the second half of the head dim into the first half (with sign flip)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.size(-2)
        # match q's dtype/device so flash attention stays enabled under autocast
        cos = self.cos_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # rope is opt-in via config; getattr keeps older configs (without these
        # fields) working as a plain gpt-2 attention with absolute pos emb.
        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            rope_base = getattr(config, "rope_base", 10000.0)
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rope_base,
                max_position_embeddings=config.block_size,
            )
        else:
            self.rotary_emb = None

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # apply rope to q and k before attention; v is not rotated
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, seq_len=T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # default to gelu so older configs (without mlp_type) are unchanged
        self.mlp_type = getattr(config, "mlp_type", "gelu")

        if self.mlp_type == "swiglu":
            # pick inner dim ~ (8/3) * n_embd so total mlp params roughly match
            # the standard 4*n_embd gelu mlp (swiglu has 3 matrices vs 2).
            inner_dim = int(4 * config.n_embd * 2 / 3)
            # round up to a multiple of 256 for better gpu kernel efficiency
            inner_dim = ((inner_dim + 255) // 256) * 256
            self.inner_dim = inner_dim
            # one fused projection produces both the gate and the up branches
            self.c_fc = nn.Linear(config.n_embd, 2 * inner_dim)
            self.c_proj = nn.Linear(inner_dim, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        if self.mlp_type == "swiglu":
            # split fused projection into (gate, up); swiglu = silu(gate) * up
            x_in = self.c_fc(x)
            x_gate, x_up = x_in.chunk(2, dim=-1)
            x = F.silu(x_gate) * x_up
            x = self.c_proj(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 
    vocab_size: int = 50257
    n_layer: int = 8  
    n_head: int = 12
    n_embd: int = 768

    # architecture toggles for rope + swiglu. defaults match a fresh run;
    # from_pretrained() overrides them so hugging face gpt-2 weights still load.
    use_rope: bool = True
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"  # "swiglu" or "gelu"


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # rope replaces absolute position embeddings, so skip wpe when use_rope=true
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        )
        if not getattr(config, "use_rope", False):
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token (and optionally absolute position) embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if getattr(self.config, "use_rope", False):
            # rope is applied inside attention, so no absolute pos emb here
            x = tok_emb
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
            pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
            x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # huggingface gpt-2 has absolute pos emb + gelu mlp, so disable rope/swiglu
        # otherwise the state-dict shapes won't line up with the hf checkpoint
        config_args["use_rope"] = False
        config_args["mlp_type"] = "gelu"
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        #     print(
        #         f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        #     )
        #     print(
        #         f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        #     )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

class NanoGPTAdapter(torch.nn.Module):
    """
    A wrapper class to make Andrej's model play nicely with the eval script.
    It strips the loss from the output and slices the padded vocabulary.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # Andrej's forward pass returns (logits, loss). We only want logits.
        logits, _ = self.model(input_ids)
        
        # Andrej pads the vocab to 50304 for TensorCore efficiency. 
        # The eval script expects exactly 50257. We slice it here.
        return logits[:, :, :50257]

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    __main__.GPTConfig = GPTConfig
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = GPTConfig(vocab_size=50304)
    
    # Initialize the raw architecture
    model = GPT(config)
    
    # Clean up the state dictionary keys
    # (Removes the '_orig_mod.' prefix if the model was compiled)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # Load the weights into the model
    model.load_state_dict(state_dict)
    
    # Wrap the model in our adapter
    wrapped_model = NanoGPTAdapter(model)
    
    # Send to GPU and set to evaluation (read-only) mode
    wrapped_model.to(device)
    wrapped_model.eval()
    
    return wrapped_model