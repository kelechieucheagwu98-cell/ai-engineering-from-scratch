import numpy as np


class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total


def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens


def layernorm_backward(dy, x, ln):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + ln.eps)
    x_hat = (x - mean) * std_inv
    n = x.shape[-1]

    grad_gamma = (dy * x_hat).sum(axis=(0, 1))
    grad_beta = dy.sum(axis=(0, 1))

    dx_hat = dy * ln.gamma
    dvar = (dx_hat * (x - mean) * -0.5 * std_inv ** 3).sum(axis=-1, keepdims=True)
    dmean = (-dx_hat * std_inv).sum(axis=-1, keepdims=True)
    dmean += dvar * (-2.0 / n) * (x - mean).sum(axis=-1, keepdims=True)
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mean) / n + dmean / n

    return dx, grad_gamma, grad_beta


def ffn_backward(dy, x_in, ffn):
    h = x_in @ ffn.W1 + ffn.b1
    h_relu = np.maximum(0, h)

    grad_W2 = h_relu.reshape(-1, h_relu.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    grad_b2 = dy.reshape(-1, dy.shape[-1]).sum(axis=0)

    dh_relu = dy @ ffn.W2.T
    dh = dh_relu * (h > 0).astype(float)

    grad_W1 = x_in.reshape(-1, x_in.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
    grad_b1 = dh.reshape(-1, dh.shape[-1]).sum(axis=0)

    dx = dh @ ffn.W1.T

    return dx, grad_W1, grad_b1, grad_W2, grad_b2


def attention_backward(dy, x_in, attn, mask):
    batch, seq_len, d = x_in.shape
    h, hd = attn.num_heads, attn.head_dim

    def split(t):
        return t.reshape(batch, seq_len, h, hd).transpose(0, 2, 1, 3)

    def merge(t):
        return t.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)

    Q, K, V = split(x_in @ attn.W_q), split(x_in @ attn.W_k), split(x_in @ attn.W_v)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    attn_out = merge(weights @ V)

    flat = lambda t: t.reshape(-1, d)
    grad_W_out = flat(attn_out).T @ flat(dy)
    d_out = split(dy @ attn.W_out.T)

    d_weights = d_out @ V.transpose(0, 1, 3, 2)
    dV = weights.transpose(0, 1, 3, 2) @ d_out
    d_scores = weights * (d_weights - (d_weights * weights).sum(axis=-1, keepdims=True))
    d_scores /= np.sqrt(hd)
    dQ = merge(d_scores @ K)
    dK = merge(d_scores.transpose(0, 1, 3, 2) @ Q)
    dV = merge(dV)

    grad_W_q = flat(x_in).T @ flat(dQ)
    grad_W_k = flat(x_in).T @ flat(dK)
    grad_W_v = flat(x_in).T @ flat(dV)
    dx = dQ @ attn.W_q.T + dK @ attn.W_k.T + dV @ attn.W_v.T

    return dx, grad_W_q, grad_W_k, grad_W_v, grad_W_out


def parameters(model):
    params = [model.embedding.token_embed, model.embedding.pos_embed]
    for block in model.blocks:
        params += [block.ln1.gamma, block.ln1.beta,
                   block.attn.W_q, block.attn.W_k, block.attn.W_v, block.attn.W_out,
                   block.ln2.gamma, block.ln2.beta,
                   block.ffn.W1, block.ffn.b1, block.ffn.W2, block.ffn.b2]
    params += [model.ln_f.gamma, model.ln_f.beta]
    return params


def loss_and_grads(model, input_ids, target_ids):
    seq_len = input_ids.shape[-1]
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    x = model.embedding.forward(input_ids)
    cache = []
    for block in model.blocks:
        ln1_out = block.ln1.forward(x)
        x_mid = x + block.attn.forward(ln1_out, mask)
        ln2_out = block.ln2.forward(x_mid)
        cache.append((x, ln1_out, x_mid, ln2_out))
        x = x_mid + block.ffn.forward(ln2_out)
    x_normed = model.ln_f.forward(x)
    logits = x_normed @ model.embedding.token_embed.T
    loss = cross_entropy_loss(logits, target_ids)
    batch_size, s_len, _ = logits.shape
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    dlogits = probs
    dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0
    dlogits /= (batch_size * s_len)
    grad_token_embed = dlogits.reshape(-1, dlogits.shape[-1]).T @ x_normed.reshape(-1, x_normed.shape[-1])
    dx_normed = dlogits @ model.embedding.token_embed
    dx, g_lnf_gamma, g_lnf_beta = layernorm_backward(dx_normed, x, model.ln_f)

    block_grads = []
    for block, (x_in, ln1_out, x_mid, ln2_out) in reversed(list(zip(model.blocks, cache))):
        d_ffn_in, gW1, gb1, gW2, gb2 = ffn_backward(dx, ln2_out, block.ffn)
        d_mid, g_ln2_gamma, g_ln2_beta = layernorm_backward(d_ffn_in, x_mid, block.ln2)
        d_mid = dx + d_mid
        d_attn_in, gWq, gWk, gWv, gWo = attention_backward(d_mid, ln1_out, block.attn, mask)
        d_in, g_ln1_gamma, g_ln1_beta = layernorm_backward(d_attn_in, x_in, block.ln1)
        dx = d_mid + d_in

        block_grads.append([g_ln1_gamma, g_ln1_beta, gWq, gWk, gWv, gWo,
                            g_ln2_gamma, g_ln2_beta, gW1, gb1, gW2, gb2])
    np.add.at(grad_token_embed, input_ids.reshape(-1), dx.reshape(-1, dx.shape[-1]))
    grad_pos_embed = np.zeros_like(model.embedding.pos_embed)
    grad_pos_embed[:seq_len] = dx.sum(axis=0)

    grads = [grad_token_embed, grad_pos_embed]
    for g in reversed(block_grads):
        grads += g
    grads += [g_lnf_gamma, g_lnf_beta]
    return loss, grads


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=1e-3):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )
    optimizer = Adam(parameters(model), lr=lr)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print(f"Random-guess loss: ln({vocab_size}) = {np.log(vocab_size):.4f}")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        loss, grads = loss_and_grads(model, input_ids, target_ids)
        optimizer.step(grads)

        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model


def parameter_breakdown():
    configs = [
        ("GPT-2 Small", 50257, 768, 12, 12, 1024, 3072),
        ("GPT-2 Medium", 50257, 1024, 16, 24, 1024, 4096),
        ("GPT-2 Large", 50257, 1280, 20, 36, 1024, 5120),
        ("GPT-2 XL", 50257, 1600, 25, 48, 1024, 6400),
    ]

    print("GPT-2 Family Parameter Counts")
    print("=" * 65)
    print(f"{'Model':<16} {'Layers':>6} {'Heads':>6} {'Dims':>6} {'Params':>14}")
    print("-" * 65)

    for name, vocab, dim, heads, layers, seq_len, ff in configs:
        token_emb = vocab * dim
        pos_emb = seq_len * dim
        per_block_attn = 4 * dim * dim
        per_block_ff = 2 * dim * ff + dim + ff
        per_block_ln = 4 * dim
        per_block = per_block_attn + per_block_ff + per_block_ln
        final_ln = 2 * dim
        total = token_emb + pos_emb + layers * per_block + final_ln
        print(f"{name:<16} {layers:>6} {heads:>6} {dim:>6} {total:>14,}")

    print()


def memory_estimate():
    print("Memory Requirements for Inference (FP16)")
    print("=" * 65)

    models = [
        ("GPT-2 Small (124M)", 124e6, 12, 12, 64, 1024),
        ("Llama 3 8B", 8e9, 32, 32, 128, 8192),
        ("Llama 3 70B", 70e9, 80, 64, 128, 8192),
        ("Llama 3 405B", 405e9, 126, 128, 128, 131072),
    ]

    print(f"{'Model':<24} {'Weights':>10} {'KV Cache':>12} {'Total':>10}")
    print("-" * 65)

    for name, params, layers, heads, head_dim, max_seq in models:
        weight_bytes = params * 2
        kv_per_token = 2 * layers * heads * head_dim * 2
        kv_full = kv_per_token * max_seq
        total = weight_bytes + kv_full

        def fmt(b):
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            return f"{b / 1e6:.0f} MB"

        print(f"{name:<24} {fmt(weight_bytes):>10} {fmt(kv_full):>12} {fmt(total):>10}")

    print()


if __name__ == "__main__":
    np.random.seed(42)

    parameter_breakdown()
    memory_estimate()

    corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

    print("Training Mini GPT")
    print("=" * 65)
    model = train_mini_gpt(corpus, num_steps=200)

    prompt = list("The transformer".encode("utf-8"))
    print(f"\nPrompt: 'The transformer'")
    print("Generating...")
    output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
    generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
    print(f"Generated: {generated_text}")
