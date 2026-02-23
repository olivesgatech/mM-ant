import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
from einops import repeat, rearrange
from model.extras.transformer import Transformer
from model.extras.position import PositionalEncoding

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class FUTR(nn.Module):
    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, query_num=49):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.query_pad_idx = query_num - 1
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(args.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.l3_attention = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        
        # New Projector for VLM Context (CLIP dim 512 -> hidden_dim)
        self.context_projector = nn.Linear(512, hidden_dim)
        nn.init.xavier_uniform_(self.context_projector.weight)

        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc_seg.weight)

        if args.anticipate :
            self.fc = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)

        self.fc_l3 = nn.Linear(hidden_dim, query_num)

        max_seq_len = args.max_pos_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_embedding)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.positional_embedding_l3 = self.sinusoidal_positional_encoding(max_seq_len, hidden_dim)
        self.positional_embedding_l3 = self.positional_embedding_l3.to(self.device)

        if args.input_type =='gt':
            self.gt_emb = nn.Embedding(n_class+2, self.hidden_dim, padding_idx=n_class+1)
            nn.init.xavier_uniform_(self.gt_emb.weight)

    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        return pos_embed

    def forward(self, inputs, query=None, context=None, mode='train', epoch=0, idx=0):
        # inputs: visual features
        # context: (Optional) Fine-grained embeddings from VLM [B, 512]
        
        if mode == 'train' :
            src, src_label = inputs
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
        else :
            src = inputs
            src_key_padding_mask = None

        tgt_mask = None
        if self.args.input_type == 'i3d_transcript':
            B, S, C = src.size()
            src = self.input_embed(src) 
        elif self.args.input_type == 'gt':
            B, S = src.size()
            src = self.gt_emb(src)
        src = F.relu(src)
        
        src = self.pos_enc(src)
       
        pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0)
        pos_embed_l3 = pos_embed_l3[:, :S,]
        
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        src = rearrange(src, 'b t c -> t b c')

        # src_l3, _ = self.l3_attention(src, src, src) 
        # src_l3 = rearrange(src_l3, 't b c -> b t c')
        # l3_logits = pos_embed_l3.to(self.device) + src_l3

        # # Base Action Query from Visual Features
        # action_query = l3_logits
        # action_query = F.adaptive_avg_pool1d(action_query.permute(0, 2, 1), self.n_query).permute(0, 2, 1) # [B, n_query, hidden]
        
        # Inject VLM Context (Fine-grained Label) if provided
        if context is not None:
            ctx_emb = self.context_projector(context)
            action_query = rearrange(ctx_emb, 'b t c -> t b c')
            tgt = torch.zeros_like(action_query)
        else:
            #tgt = rearrange(src, 't b c -> b t c')
            tgt = torch.zeros_like(src)
            action_query = None
        
        pos = rearrange(pos, 'b t c -> t b c')
        
        src, tgt = self.transformer(src=src, tgt=tgt, mask=src_key_padding_mask, tgt_mask=tgt_mask, 
                                    tgt_key_padding_mask=None, query_embed=action_query, 
                                    pos_embed=pos, tgt_pos_embed=None, epoch=epoch, idx=idx)
        output = dict()
        src = rearrange(src, 't b c -> b t c')
        if tgt is None:
            tgt_seg = self.fc_seg(src)
            output['seg'] = tgt_seg
            output['action'] = None
            output['duration'] = None
            return output

        tgt = rearrange(tgt, 't b c -> b t c')
        
        if self.args.anticipate :
            output_class = self.fc(tgt) 
            duration = self.fc_len(tgt)
            duration = duration.squeeze(2)
            output['duration'] = duration
            output['action'] = output_class

        if self.args.seg :
            tgt_seg = self.fc_seg(src)
            output['seg'] = tgt_seg

        return output

def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)