import torch
from torch import nn, LongTensor
import math
import dgl
from dgl import function as fn
from torch.nn import functional as F
from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.ops import gather_mm
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class RGCN(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=3, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, edge_num=21):
        super().__init__()
        config = Config(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, edge_num=edge_num)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.obj_emb = nn.Embedding(config.vocab_size, 1024)
        self.edge_emb = nn.Embedding(edge_num, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[RGATConv(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, objs, graph, targets=None):

        objs_emb = self.obj_emb(objs)
        x = objs_emb

        b, t, d = x.shape[0], x.shape[1], x.shape[2]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(x + position_embeddings)
        x = x.view(-1, d)
        efeat = self.edge_emb(graph.edata['type'])
        for block in self.blocks:
            x, efeat = block(graph, x, efeat)

        x = self.ln_f(x)
        logits = self.head(x)

        return x, logits

class RGCNConv(nn.Module):
    def __init__(self, config):
        super(RGCNConv, self).__init__()

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.eW = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.n_head = config.n_head
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.W = nn.Parameter(torch.Tensor(config.edge_num, config.n_embd, config.n_embd))
        self.config = config
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.W, -1 / math.sqrt(self.config.n_embd), 1 / math.sqrt(self.config.n_embd))


    def get_scocre(self, edges):
        return {'score': (edges.dst['q'] * edges.src['k']).sum(-1, keepdim=True)}

    def scaled_exp(self, edges):
        return {'score': (edges.data['score'] / math.sqrt(edges.src['k'].size(-1))).clamp(-10, 10)}


    def get_qkv(self, edges):
        return {'qkv': edges.data['score'] @ edges.src['v']}

    def message(self, edges):
        B = edges.src['v'].shape[0]
        m = gather_mm(edges.src['v'], self.W, idx_b=edges.data['type'])
        m = (m - edges.data['efeat']).view(B, self.n_head, -1)
        m = m * edges.data['score']
        return {'m' : m}

    def message_tmp(self, edges):
        B = edges.src['v'].shape[0]
        m = gather_mm(edges.src['v'], self.W, idx_b=edges.data['type'])
        m = (m - edges.data['efeat']).view(B, self.n_head, -1)
        score = self.a * F.leaky_relu(m)
        m = m * score.sum(dim=-1).unsqueeze(-1)
        return {'m' : m}

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            B, C = feat.size()
            E, _ = efeat.size()
            graph.dstdata['q'] = self.query(feat).view(B, self.n_head, C // self.n_head)
            graph.srcdata['k'] = self.key(feat).view(B, self.n_head, C // self.n_head)
            graph.srcdata['v'] = self.value(feat)

            graph.apply_edges(self.get_scocre)
            graph.apply_edges(self.scaled_exp)
            graph.edata['efeat'] = efeat
            graph.edata['score'] = edge_softmax(graph, graph.edata['score'])


            graph.update_all(self.message, fn.sum('m', 'wv'))

            attn = graph.dstdata['wv'].view(B, C)
            y = self.proj(attn)
        efeat = self.eW(efeat)
        feat = feat + y
        feat = feat + self.mlp(self.ln2(feat))
        return feat, efeat

