import torch
from torch import nn, LongTensor
import math
import dgl
from dgl import function as fn
from torch.nn import functional as F
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.base import DGLError


class Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GAT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=3, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = Config(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.obj_emb = nn.Embedding(config.vocab_size, 1024)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[GptGraphConv(config) for _ in range(config.n_layer)])
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


    def forward(self, z_indices, objs, poss, graph, targets=None):
        n_obj = 9
        objs_emb = self.obj_emb(objs)
        poss_emb = self.poss_emb(poss)
        poss_emb = torch.split(poss_emb, n_obj, dim=1)
        nfeat = torch.cat([objs_emb, poss_emb[0], poss_emb[1]], dim=-1) # b, 9, d

        z = self.tok_emb(z_indices)

        x = torch.cat([nfeat, z], dim=1)
        b, t, d = x.shape[0], x.shape[1], x.shape[2]
        if t < self.block_size:
            index = LongTensor(list(range(t)) * b).view(b, -1)
            index = [index[idx] + idx * 265 for idx in range(b)]
            index = torch.cat(index).to(graph.device)

            graph = dgl.node_subgraph(graph, index)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(x + position_embeddings)
        x = x.view(-1, d)
        for block in self.blocks:
            x = block(graph, x)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        logits = logits.view(b, t, -1)
        x = x.view(b, t, d)
        return logits, loss, x[:, :n_obj, :]

class GATConv(nn.Module):
    def __init__(self, config):
        super(GATConv, self).__init__()

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.n_head = config.n_head
        self.proj = nn.Linear(config.n_embd, config.n_embd)


    def get_scocre(self, edges):
        return {'score': (edges.dst['q'] * edges.src['k']).sum(-1, keepdim=True)}

    def scaled_exp(self, edges):
        return {'score': (edges.data['score'] / math.sqrt(edges.src['k'].size(-1))).clamp(-10, 10)}

    def get_qkv(self, edges):
        return {'qkv': edges.data['score'] @ edges.src['v']}

    def forward(self, graph, feat):
        with graph.local_scope():
            B, C = feat.size()
            graph.dstdata['q'] = self.query(feat).view(B, self.n_head, C // self.n_head)
            graph.srcdata['k'] = self.key(feat).view(B, self.n_head, C // self.n_head)
            graph.srcdata['v'] = self.value(feat).view(B, self.n_head, C // self.n_head)
            graph.apply_edges(self.get_scocre)
            graph.apply_edges(self.scaled_exp)
            graph.edata['score'] = edge_softmax(graph, graph.edata['score'])
            graph.update_all(fn.src_mul_edge('v', 'score', 'm'), fn.sum('m', 'wv'))

            attn = graph.dstdata['wv'].view(B, C)
            y = self.proj(attn)

        feat = feat + y
        feat = feat + self.mlp(self.ln2(feat))
        return feat


