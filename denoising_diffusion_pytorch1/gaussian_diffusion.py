import torch
from torch import nn
import torch.nn.functional as F
from denoising_diffusion_pytorch1.lstm_diffusion import LSTM_with_timeemb
from denoising_diffusion_pytorch1.encoder import Encoder
from denoising_diffusion_pytorch import Unet
from tqdm.auto import tqdm
import math
from einops import rearrange, reduce
from collections import namedtuple
from functools import partial


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 timesteps,   #(500,4,32, vocab_size,64,128,pred_x0)
                 max_sent,
                 max_word,
                 vocab_size,
                 words_emb_dim,
                 hidden_dim,
                 pred_method,
                 beta_schedule='cosine',
                 p2_loss_weight_k=1,
                 p2_loss_weight_gamma=0,
                 loss_type='l1'):
        super(GaussianDiffusion, self).__init__()


        self.max_sent = max_sent
        self.max_word = max_word
        self.words_emb_dim = words_emb_dim  #(64)
        self.vocab_size = vocab_size
        self.loss_type = loss_type
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.model = LSTM_with_timeemb(max_sent, max_word, hidden_dim, words_emb_dim, hidden_dim)   #(4,32,128,64,128)
        self.objective = pred_method
        # self.model = Unet(dim=64, channels=num_sent, dim_mults=(1, 2, 4))
        self.num_timesteps = timesteps
        self.encoder = Encoder(words_emb_dim, 7)
        self.emb = Embeddings(vocab_size, words_emb_dim)
        # self.anti_emb = AntiEmbeddings(vocab_size, words_emb_dim)
        #1113
        num_heads = 8
        num_layers = 6
        # 定义 Transformer 解码器
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=words_emb_dim,  # 输入的嵌入维度，通常是隐藏维度
            nhead=num_heads  # 多头注意力机制的头数
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,  # 使用定义好的 Transformer 解码器层
            num_layers=num_layers  # Transformer 的层数
        )


        self.anti_emb = AntiEmbeddings(vocab_size, words_emb_dim, num_heads, num_layers)

        #到这
        self.criterion = nn.CrossEntropyLoss()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, t, captions, topic, noise=None, loss_type="l1", train='emb'):#topic图片captions词

        if train == 'emb':
            x_start = self.emb(captions)   #原始文本x0
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  #xt
            with torch.no_grad():
                model_out = self.model(x_noisy, t, topic)
        elif train == 'dm':
            with torch.no_grad():
                x_start = self.emb(captions)
                #print('x_start.shapex_start.shapex_start.shapex_start.shapex_start.shape',x_start.shape)   #batchsize, max_sent*max_word, words_emb_dim
                #torch.Size([64, 128, 256])
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_out = self.model(x_noisy, t, topic)
            #print("形状形状笑笑笑笑笑笑笑笑笑","x",x_noisy.shape,"t",t.shape,"topic",topic.shape)#x torch.Size([64, 128, 256]) t torch.Size([64]) topic torch.Size([64, 256])
            #print("model_out",model_out.shape)#torch.Size([64, 128, 256])

        if self.objective == 'pred_noise':
            target = noise
            print('pred_noisetarget.shape|||||||||||||||||||pred_noisetarget.shape', target.shape)
        elif self.objective == 'pred_x0':
            target = x_start
            # print('pred_x0target.shape|||||||||||||||||||pred_x0target.shape',target.shape)    #batchsize, max_sent*max_word, words_emb_dim
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # if train == 'dm':
        #     with torch.no_grad():
        #         words_out = self.anti_emb(x_start)[0]  # (b,n,l,d)
        # else:
        #     words_out = self.anti_emb(x_start)[0]
        words_out = self.anti_emb(x_start)[0]
        # print('words_out:',words_out.shape)   #batchsize, max_sent*max_word, vocab_size
        loss_ae = self.criterion(words_out.view(-1, self.vocab_size), captions.view(-1))

        if train == 'dm':
            # print('model_out.shape,target.shape||||||||||||||||||||||||||model_out.shape,target.shape',model_out.shape,target.shape)#batchsize, max_sent*max_word, words_emb_dim
            # print(f"x_start.shape: {x_start.shape}")#batchsize, max_sent*max_word, words_emb_dim
            # print(f"noise.shape: {noise.shape}")#batchsize, max_sent*max_word, words_emb_dim
            # print(f"target.shape: {target.shape}, target.numel: {target.numel()}")#batchsize, max_sent*max_word, words_emb_dim      #batchsize*max_sent*max_word*words_emb_dim

            # target = target.view(16, 4, 32, 64)  # 调整为 [16, 4, 32, 64]

            loss_noisy = self.loss_fn(model_out, target, reduction='none')
            loss_noisy = reduce(loss_noisy, 'b ... -> b (...)', 'mean')
            loss_noisy = loss_noisy * extract(self.p2_loss_weight, t, loss_noisy.shape)
            loss_noisy = loss_noisy.mean()
            return loss_noisy, loss_ae
        elif train == 'emb':
            # loss_emb = self.criterion(words_out.view(-1, self.vocab_size), captions.view(-1))
            loss_emb = self.loss_fn(model_out, target)
            return loss_emb, loss_ae

    def forward(self, captions, img):
        batch_size, device = captions.shape[0], captions.device  # x0(batchsize,n,l)
        # print('captions.shape:',captions.shape)   #(16,4,32)
        # captions = x.clone()
        t_dm = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)    #生成16个0到时间步1000之间的数
        # print(t_dm,'t_dm')

        t_emb = torch.full((batch_size,), 1, device=device, dtype=torch.long)      #生成16个1
        # x = self.emb(x)
        # print(t_emb)
        # print('img.shape::::',img.shape)   #16,3,256,256    最后一个epoch只有14了,已修复
        topic = self.encoder(img)       #图像的特征向量
        # print(topic.shape)    #(16,64,64)
        # print('198198198198198198',captions.shape)

        return self.train_mode('dm', t_emb, t_dm, captions, topic)   #return self.train_mode('dm', t_emb, t_dm, 文本, 特征向量)

    def train_mode(self, mode, t_emb, t_dm, captions, topic):


        #
        loss_emb, loss_ae2 = self.p_losses(t_emb, captions, topic, train='emb') if mode == 'emb' else (0., 0.)
        # print('captions.shape',captions.shape)    #torch.Size([batchsize, max_sent, max_word])


        loss_dm, loss_ae1 = self.p_losses(t_dm, captions, topic, train='dm') if mode == 'dm' else (0., 0.)


        if mode == 'emb':
            loss = loss_emb + loss_ae2
            loss_dm_i = loss_dm
            loss_emb_i = loss_emb.item()
            loss_ae = loss_ae1 + loss_ae2.item()
        elif mode == 'dm':
            loss = loss_dm + loss_ae1
            loss_dm_i = loss_dm.item()
            loss_emb_i = loss_emb
            loss_ae = loss_ae1.item() + loss_ae2

        return loss, loss_dm_i, loss_emb_i, loss_ae

    def predict_start_from_noise(self, x_t, t, noise):   #根据噪声预测x_0
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):       #根据x_0预测噪声
        # print('x_t.shape,x0.shape|||||||||||||x_t.shape,x0.shape',x_t.shape, x0.shape)  #torch.Size([batchsize, max_sent*max_word, words_emb_dim])
        x_t = x_t.view_as(x0)  # 将 x0 的形状调整为和 x_t 一致

        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, topic=None):
        model_output = self.model(x, t, topic)
        # print('x.shape|||||||||||||||||||||||x.shape',x.shape)      #torch.Size([batchsize, max_sent*max_word, words_emb_dim])
        # print('model_output.shape|||||||||||||||||||||||||model_output.shape',model_output.shape)     #torch.Size([batchsize, max_sent*max_word, words_emb_dim])
        # model_output = model_output.view(16, self.max_sent*self.max_word, self.words_emb_dim)  # 调整为 [16, 4, 32, 64]
        # print('x.shape|||||||||||||||||||||||x.shape', x.shape)  # torch.Size([16, 128, 64])


        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)


        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, topic=None):   #计算的是反向扩散过程中的模型均值 (`model_mean`)，后验方差 (`posterior_variance`)，后验对数方差 (`posterior_log_variance`) 和原始数据x0。这些值用于指导采样过程。
        preds = self.model_predictions(x, t, topic)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_posterior(self, x_start, x_t, t): #计算的是扩散过程中的后验均值、方差和对数方差。这些值是基于当前时间步 `t`、去噪的图像 `x_start` 和当前图像 `x_t` 计算的。
        # 如果 x_t 是 (batch_size, 128, 64)，需要调整到 (batch_size, 4, 32, 64)
        x_t = x_t.view(x_t.shape[0], self.max_word*self.max_sent, x_t.shape[-1])

        # print(f"x_start.shape: {x_start.shape}, x_t.shape: {x_t.shape}, t.shape: {t.shape}")   #torch.Size([batchsize, max_sent*max_word, words_emb_dim])    t:16
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t, topic):    #单步逆扩散过程，从时间步 t 转移到时间步 t-1。
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, topic=topic)
        # noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        noise = torch.randn_like(model_mean)


        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img   #这是通过反向扩散过程从噪声图像生成的去噪后图像

    @torch.no_grad()
    def p_sample_loop(self, shape, topic):
        b = shape[0]
        # xt = torch.randn(shape, device=topic.device)
        w = torch.randint(0, self.vocab_size, shape, device=topic.device)
        # print('w::::',w.shape)
        xt = self.emb(w)
        # print('xt.shape',xt.shape)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            xt = self.p_sample(xt, t, topic)

        xt = self.anti_emb(xt)
        return xt[1]

    @torch.no_grad()
    def sample(self, topic, batch_size=16):
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, self.max_sent, self.max_word), topic)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, words_emb_dim):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, words_emb_dim)
        self.norm = nn.LayerNorm(words_emb_dim)
        self.word_emb = nn.Sequential(
            nn.Linear(words_emb_dim, words_emb_dim),
            nn.GELU(),
            nn.Linear(words_emb_dim, words_emb_dim)
        )

    def forward(self, x):
        # x: [batch_size, num_sentences, seq_len]
        x = self.emb(x)  # [batch_size, num_sentences, seq_len, words_emb_dim]
        x = self.norm(x)
        x = self.word_emb(x)  # 形状未变
        # 调整为 [batch_size, num_sentences * seq_len, words_emb_dim]
        batch_size, num_sentences, seq_len, emb_dim = x.size()
        return x.view(batch_size, num_sentences * seq_len, emb_dim)




# class AntiEmbeddings(nn.Module):
#     def __init__(self, vocab_size, words_emb_dim):
#         super(AntiEmbeddings, self).__init__()
#         self.anti_emb = nn.Sequential(
#             nn.Linear(words_emb_dim, words_emb_dim),
#             nn.GELU(),
#             nn.Linear(words_emb_dim, vocab_size)
#         )
#
#     def forward(self, x):
#         x = self.anti_emb(x)
#         pred = torch.max(x, dim=-1)[1]
#         print('pred::::::::::::::::',pred)
#         return x, pred



#1113加的

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class AntiEmbeddings(nn.Module):
    def __init__(self, vocab_size, words_emb_dim, num_heads, num_layers, max_seq_len=500):
        super(AntiEmbeddings, self).__init__()

        # 词汇嵌入层
        self.embedding = nn.Embedding(vocab_size, words_emb_dim)

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, words_emb_dim))

        # 定义单层解码器层，这里直接用 words_emb_dim
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=words_emb_dim, nhead=num_heads)

        # 定义多层解码器
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # 输出层，将解码结果映射到词汇表大小
        self.fc_out = nn.Linear(words_emb_dim, vocab_size)

    def forward(self, x):
        # 假设 x 是 3 维 (batch_size, seq_len, emb_dim)
        # print('Input shape:', x.size())  #batchsize, max_sent*max_word, words_emb_dim
        if len(x.size()) == 4:
            # 如果输入为 4 维，先降为 3 维
            batch_size, max_sen, seq_len, emb_dim = x.size()
            x = x.view(batch_size, max_sen*seq_len, emb_dim)

        # 添加位置编码
        seq_len = x.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :]
        # print('Positional encoding shape:', positional_encoding.shape, 'Input shape:', x.size()) #1,128,128       #batchsize, max_sent*max_word, words_emb_dim
        x = x + positional_encoding

        # 使用 Transformer 解码器
        transformer_output = self.transformer_decoder(x, x)  # 假设 x 是目标嵌入

        # 通过线性层输出词汇的 logits
        logits = self.fc_out(transformer_output)

        # 计算预测的词
        pred = torch.argmax(logits, dim=-1)

        return logits, pred





#
# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, words_emb_dim, nhead=8, num_layers=6, dim_feedforward=2048):
#         super(TransformerDecoder, self).__init__()
#
#         self.embedding = nn.Embedding(vocab_size, words_emb_dim)  # 词汇嵌入层
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, words_emb_dim))  # 位置编码
#         self.decoder_layer = TransformerDecoderLayer(d_model=words_emb_dim, nhead=nhead, dim_feedforward=dim_feedforward)  # 使用自定义解码器层
#         self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)  # num_layers 传给 TransformerDecoder
#         self.output_layer = nn.Linear(words_emb_dim, vocab_size)  # 词汇映射层
#
#     def forward(self, latent_vector, target_sequence=None):
#         # latent_vector 是来自扩散模型的潜在空间向量
#         batch_size = latent_vector.size(0)
#         seq_len = 10  # 可以根据需要修改，表示目标序列的最大长度
#
#         # 给定一个随机生成的目标序列，作为解码器的输入
#         if target_sequence is None:
#             target_sequence = torch.zeros(batch_size, seq_len).long().to(latent_vector.device)  # (batch_size, seq_len)
#
#         # 嵌入目标序列
#         target_emb = self.embedding(target_sequence) + self.positional_encoding[:, :seq_len, :]
#
#         # 扩展潜在向量的维度，适应 Transformer 输入的格式
#         latent_vector = latent_vector.unsqueeze(0).repeat(seq_len, 1, 1)  # (seq_len, batch_size, words_emb_dim)
#
#         # 通过 Transformer 解码器
#         output = self.transformer_decoder(target_emb, latent_vector)
#
#         # 将解码器的输出通过线性层映射回词汇表大小
#         logits = self.output_layer(output)  # (seq_len, batch_size, vocab_size)
#
#         return logits
#
#
# def generate_text(transformer_model, latent_vector, max_word=20, temperature=1.0):
#     batch_size = latent_vector.size(0)
#     generated_sequence = torch.zeros(batch_size, max_word).long().to(latent_vector.device)
#
#     for t in range(max_word):
#         logits = transformer_model(latent_vector, target_sequence=generated_sequence[:, :t + 1])  # 生成下一个词
#         logits = logits[-1] / temperature  # 温度采样（可以进行调节）
#         prob = F.softmax(logits, dim=-1)
#
#         # 选择最大概率的词
#         next_word = torch.argmax(prob, dim=-1)  # 选择最大概率的词
#         generated_sequence[:, t] = next_word  # 更新生成序列
#
#     return generated_sequence
#
#
# #TransformerDecoderLayer##############################################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm
# from typing import Callable, Union, Optional
#
# class TransformerDecoderLayer(Module):
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerDecoderLayer, self).__init__()
#
#         # Self-attention mechanism (multi-head attention for self interactions)
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
#
#         # Cross-attention mechanism (attention to encoder's output)
#         self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
#
#         # Feedforward network (position-wise)
#         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)  # First linear transformation
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)  # Second linear transformation to return to d_model size
#
#         # Normalization layers
#         self.norm_first = norm_first
#         self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#
#         # Dropout layers
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#         self.dropout3 = Dropout(dropout)
#
#         # Activation function (default to ReLU or GELU if specified)
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerDecoderLayer, self).__setstate__(state)
#
#     def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
#                 memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
#                 memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Pass the inputs (and mask) through the decoder layer.
#
#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).
#
#         Shape:
#             - tgt: (batch_size, seq_len, d_model)
#             - memory: (batch_size, seq_len, d_model)
#             - tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask: (optional masks)
#         """
#         x = tgt
#
#         if self.norm_first:
#             # Apply residual connection followed by normalization
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
#             x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#             x = x + self._ff_block(self.norm3(x))
#         else:
#             # Apply normalization first and then residual connection
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
#             x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             x = self.norm3(x + self._ff_block(x))
#
#         return x
#
#     def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         """ Self-attention block (applies multihead attention to the target sequence). """
#         x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
#         return self.dropout1(x)
#
#     def _mha_block(self, x: torch.Tensor, mem: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         """ Multi-head attention block (applies multihead attention to the memory and target sequence). """
#         x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
#         return self.dropout2(x)
#
#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         """ Feedforward block (applies linear layers and activation function). """
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)
#
#
# # Helper function for activation
# def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
#     """ Return the corresponding activation function based on string input. """
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu
#     else:
#         raise ValueError(f"Activation function {activation} not supported")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #到这





if __name__ == '__main__':
    # device = torch.device('cuda')
    # torch.backends.cudnn.benchmark = False
    gd = GaussianDiffusion(100, 4, 32, 100, 128, 256, 'pred_x0', loss_type='l2')
    params_dicts = [
        {'params': gd.model.parameters(), 'lr': 1e-3},
        {'params': gd.anti_emb.parameters(), 'lr': 1e-3},
    ]
    optim_model = torch.optim.Adam(params=params_dicts)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max=200)
    print(sum([param.nelement() for param in gd.parameters()]))
    x0 = torch.randint(0, 100, [8, 4, 32])
    img = torch.randn([8, 3, 224, 224])
    for i in range(500):
        optim_model.zero_grad()
        loss, loss1, loss2 = gd(x0, img)
        # loss = gd.p_losses(model, x0, t)
        loss.backward()
        optim_model.step()
        print('epoch:%d loss: noisy(%.3f) emb(%.3f) lr: noisy(%.4f) emb(%.4f)' %
              (i, loss1, loss2, *[i['lr'] for i in optim_model.param_groups]))
        # scheduler.step()
    pred = gd.sample(8)
    # print(pred, x0)
