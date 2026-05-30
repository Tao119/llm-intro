import sys
sys.path.append("..")
import numpy as np
import math


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)
        self.eps   = eps
        self.cache = None

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        self.cache = (x, x_hat, mean, var)
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, x_hat, mean, var = self.cache
        N = x.shape[-1]
        dgamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta  = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dx_hat = dout * self.gamma
        dvar = (dx_hat * (x - mean) * -0.5 * (var + self.eps)**-1.5).sum(axis=-1, keepdims=True)
        dmean = (dx_hat * -(var + self.eps)**-0.5).sum(axis=-1, keepdims=True) + \
                dvar * (-2 * (x - mean)).sum(axis=-1, keepdims=True) / N
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2*(x - mean)/N + dmean/N
        self.gamma += dgamma
        self.beta  += dbeta
        return dx


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) / scale
        self.W_k = np.random.randn(d_model, d_model) / scale
        self.W_v = np.random.randn(d_model, d_model) / scale
        self.W_o = np.random.randn(d_model, d_model) / scale

    def split_heads(self, x):
        N, T, D = x.shape
        x = x.reshape(N, T, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        N = Q.shape[0]
        q = self.split_heads(Q @ self.W_q)
        k = self.split_heads(K @ self.W_k)
        v = self.split_heads(V @ self.W_v)

        score = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        if mask is not None:
            score += (mask * -1e9)
        attn = softmax(score)
        self._attn = attn
        context = attn @ v

        context = context.transpose(0, 2, 1, 3).reshape(N, -1, self.d_model)
        return context @ self.W_o


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1  = LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff)
        self.ln2  = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ln1.forward(x + self.attn.forward(x, x, x, mask))
        x = self.ln2.forward(x + self.ff.forward(x))
        return x


class TransformerEncoder:
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, n_layers=2, max_len=128):
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pos_enc = self._positional_encoding(max_len, d_model)
        self.blocks = [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def _positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def forward(self, x_ids, mask=None):
        x = self.embedding[x_ids] + self.pos_enc[:x_ids.shape[1]]
        for block in self.blocks:
            x = block.forward(x, mask)
        return x


if __name__ == "__main__":
    print("=== Transformer Encoder demo ===")
    vocab_size = 100
    d_model    = 64
    n_heads    = 4
    d_ff       = 256
    n_layers   = 2
    batch_size = 2
    seq_len    = 10

    model = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_layers)

    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    out = model.forward(x)
    print(f"input shape : {x.shape}")
    print(f"output shape: {out.shape}")
    print(f"output[0,0,:8]: {out[0,0,:8].round(3)}")

    print("\n=== Multi-Head Attention shape check ===")
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    Q = K = V = np.random.randn(2, seq_len, 64)
    out_attn = mha.forward(Q, K, V)
    print(f"Q/K/V shape: {Q.shape}  →  attn output: {out_attn.shape}")
    print(f"attention weights (head 0): {mha._attn[0,0].round(3)}")

    print("\n=== Positional Encoding (first 3 positions, first 8 dims) ===")
    print(model.pos_enc[:3, :8].round(3))
