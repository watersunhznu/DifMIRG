import torch
from torch import nn, einsum
from einops import rearrange, reduce
import math
import torch.nn.functional as F
from inspect import isfunction


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, topic):
        return self.fn(x, topic) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, topic):
        x = self.norm(x)
        return self.fn(x, topic)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim=None, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        ctx_dim = default(ctx_dim, dim)
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, topic=None):  # (b sw d)
        # b, s, w, d = x.shape
        # x = rearrange(x, 'b s w d -> b d (s w)')
        x = x.transpose(1, 2)
        q = self.to_q(x)
        ctx = default(topic, x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)





#1113加的
        # 假设 q, k, v 的维度为 (batch_size, seq_len, dim)
        # 如果维度是 (batch_size, dim)，可以先增加一个维度
        q, k, v = [t if t.dim() == 3 else t.unsqueeze(1) for t in (q, k, v)]
        # 然后继续进行 rearrange 操作
#加到这

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        # q, k = map(l2norm, (q, k))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        # return rearrange(self.to_out(out), 'b d (s w)->b s w d', s=s)
        return self.to_out(out).transpose(1, 2)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, max_sent, max_word, words_emb_dim, hidden_dim):
        super(BasicTransformerBlock, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.feedback = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.catt = Residual(PreNorm(hidden_dim, CrossAttention(max_sent * max_word, words_emb_dim)))

    def forward(self, pack):
        x, topic = pack
        x = self.catt(x, topic)
        x = self.catt(x, topic)
        return (self.feedback(self.layernorm(x)), topic)


class SpatialTransformer(nn.Module):
    def __init__(self, max_sent, max_word, words_emb_dim, hidden_dim, block_num):
        super(SpatialTransformer, self).__init__()
        self.transformer = nn.Sequential(
            *[BasicTransformerBlock(max_sent, max_word, words_emb_dim, hidden_dim) for _ in range(block_num)]
        )

    def forward(self, x, topic):
        return self.transformer((x, topic))[0]


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)  # 先进行一次token-mixing MLP
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)  # 再进行一次channel-mixing MLP
        # x = self.dropout(x)
        return x


class LSTM_with_timeemb(nn.Module):
    def __init__(self, max_sent, max_word, time_emb_dim, words_emb_dim, hidden_dim):  #(4,32,128,64,128)
        super(LSTM_with_timeemb, self).__init__()

        self.learned_sinusoidal_cond = None
        self.self_condition = False

        self.time_emb_dim = time_emb_dim
        sinu_pos_emb = SinusoidalPosEmb(self.time_emb_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        self.time_emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2)
        )
        self.res_catt = Residual(PreNorm(hidden_dim, CrossAttention(max_sent * max_word, words_emb_dim)))
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.o_fc = nn.Linear(words_emb_dim, words_emb_dim)
        self.transformer = SpatialTransformer(max_sent, max_word, words_emb_dim, hidden_dim, 16)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.mlp1 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.mlp2 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.fc1 = nn.Linear(words_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, words_emb_dim)

    def time_embedding(self, x, time_emb):  # (b sw d) (b d)
        time_emb = self.time_emb_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b 1 c')
        scale, shift = time_emb.chunk(2, dim=2)
        h = self.layernorm(x)
        h = h * (scale + 1) + shift
        return x + h  # (b sw d)

    def lstm_with_sum(self, x, lstm):  # (b sw d)
        # s = x.shape[1]
        # x = rearrange(x, 'b s w d -> b (s w) d')
        x, _ = lstm(x)
        # x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return x

    def forward(self, x, t, topic):  # x: [batch_size, seq_len * num_sentences, emb_dim]
        time_emb = self.time_mlp(t)  # time embedding
        # b, sw, d = x.shape
        # s = 4  # 需要预定义 num_sentences 参数
        # w = sw // s
        #
        # # 恢复为四维数据
        # x = x.view(b, s, w, d)

        # 原逻辑
        # x = rearrange(x, 'b s w d -> b (s w) d')
        x = self.fc1(x)  # word_dim -> hidden_dim
        l1 = x.clone()

        x = self.lstm_with_sum(self.time_embedding(x, time_emb), self.lstm1)  # (b sw d)
        l2 = x.clone()
        x = self.mlp1(self.res_catt(self.time_embedding(x, time_emb), topic))  # (b sw d)
        l3 = x.clone()
        x = self.transformer(self.time_embedding(x, time_emb), topic)
        x = self.mlp2(self.res_catt(self.time_embedding(x, time_emb) + l3, topic))
        x = self.lstm_with_sum(self.time_embedding(x, time_emb) + l2, self.lstm5)  # (b sw d)
        x = self.fc2(self.time_embedding(x, time_emb) + l1)
        # x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return self.o_fc(x)


# class Embeddings(nn.Module):
#     def __init__(self, vocab_size, words_emb_dim, num_sent):
#         super(Embeddings, self).__init__()
#         self.emb = nn.Embedding(vocab_size, words_emb_dim)
#         self.norm = nn.GroupNorm(1, num_sent)
#
#     def forward(self, x):
#         return self.norm(self.emb(x))


if __name__ == '__main__':
    with torch.no_grad():
        lstm = LSTM_with_timeemb(4, 32, 256, 128, 256).cuda(1)
        embeddings = nn.Embedding(512, 128).cuda(1)
        inpt = embeddings(torch.randint(0, 512, [1, 4, 32]).long().cuda(1))
        out = lstm(inpt, torch.randint(0, 1000, (1,)).long().cuda(1), torch.randn([1, 64, 128]).cuda(1))
        # att = Attention(128)
        # inp = torch.randn([16, 4, 32, 256])
        # out = att(inp)
        print(out.shape)
