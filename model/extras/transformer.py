"""
FUTR Transformer class.

Copy-paste from github.com/facebookresearch/detr/blob/main/models/transformer.py with modifications.

"""

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import copy
from typing import Optional, List
import matplotlib.pyplot as plt
import threading

import math

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_num=0, hidden_dim=0, max_seq_len=0, device=None, n_query=0):
        super().__init__()

        self.d_head = d_head = d_model // nhead

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        ###################### Embedding inside the transformer #############################
        if query_num != 0:
            self.device = device
            self.query_embed = nn.Embedding(query_num, hidden_dim)
            self.positional_embedding_l3 = self.sinusoidal_positional_encoding(max_seq_len, hidden_dim).to(self.device)

        if hidden_dim != 0:
            self.device = device
            self.positional_embedding_l3 = self.sinusoidal_positional_encoding(max_seq_len, hidden_dim).to(self.device)
            self.l3_attention = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
            self.n_query = n_query
        
        #####################################################################################

        

    def sinusoidal_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_embed = torch.zeros(seq_len, emb_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term) # apply sine to even indices
        pos_embed[:, 1::2] = torch.cos(position * div_term) # apply cosine to odd indices
        return pos_embed

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, mask, tgt_mask, tgt_key_padding_mask, query_embed=None, pos_embed=None, tgt_pos_embed=None, epoch=0, idx=0, memory_mask=None, image_path=None, human_prompt=None):

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #memory = src
        ############### L3 query with LLM ##########################

        # with torch.no_grad():
        #     #with open(f'/home/seulgi/work/darai-anticipation/FUTR_proposed/image_path_{epoch}_{idx}.txt', 'w+') as f:
        #     labels_list = []
        #     for i in range(src.size(1)):
        #         _, labels = get_fine_grained_labels(image_path[i], human_prompt[i])
        #         if len(labels) > src.size(0):
        #             labels = labels[:src.size(0)]
        #         if len(labels) < src.size(0):
        #             labels += [47] * (src.size(0) - len(labels))
        #         labels_list.append(labels)
        #     labels_list = torch.tensor(labels_list).to(self.device)
            #     f.write('##########################################')
            #     f.write('\n')
            #     f.write(human_prompt[i])
            #     f.write('\n')
            #     f.write(answer)
            #     f.write('\n')
            #     f.write(','.join(map(str, labels)))
            #     f.write('\n')
            #     f.write('##########################################')
            # f.close()

        # query_embed = self.query_embed(labels_list) #(labels: (125) -> query_embed: (125, 128))
        # _, S, _ = query_embed.size()
        # pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0) # (1, 2000, 128)
        # query_embed = pos_embed_l3[:, :S,].to(self.device) + query_embed # (1, 537, 128)
        # query_embed = rearrange(query_embed, 'b t c -> t b c')
        ###################################################


        #################### L3 query #################################
        if query_embed == None:
            return memory, None
            # src_l3, _ = self.l3_attention(memory, src, src)
            # #src_l3 = src + memory
            # src_l3 = rearrange(src_l3, 't b c -> b t c')
            
            # _, S, _ = src_l3.size()
            # pos_embed_l3 = self.positional_embedding_l3.unsqueeze(0) # (1, 2000, 128)
            # labels_list = pos_embed_l3[:, :S,].to(self.device) + src_l3 # (1, 537, 128)
            # #labels_list = labels_list + rearrange(memory, 't b c -> b t c')
            # query_embed = F.adaptive_avg_pool1d(labels_list.permute(0, 2, 1), self.n_query).permute(0, 2, 1)
            # query_embed = rearrange(query_embed, 'b t c -> t b c')
            # tgt = torch.zeros_like(query_embed)
        ###############################################################

        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed, epoch=epoch, idx=idx, memory_mask=memory_mask)
        return memory, hs#, labels_list

class TransformerEncoder(nn.Module) :

    def __init__(self, encoder_layer, num_layers, norm=None) :
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, epoch=0, idx=0):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, tgt_pos=tgt_pos, epoch=epoch, idx=idx)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(src, pos)
        src2, attn_map_post = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = v = self.with_pos_embed(src2, pos)
        src2, attn_map_pre = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] =None, epoch=0, idx=0):
        q = k = v = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        #query = self.with_pos_embed(tgt, query_pos)
        #attn_mask = (query == 1000.0).to('cuda')
        #attn_mask = attn_mask.any(dim=1).unsqueeze(0).expand(query.shape[0], -1)  # (seq_len, seq_len)

        # Mask를 -inf로 변환
        #attn_mask = attn_mask.float() * -float('inf')
        tgt2, attn_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=self.with_pos_embed(memory, pos),
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        ##################### ATTENTION MAP VISUALIZATION ######
        # if threading.current_thread().name == 'MainThread':
        #     num_heads, t, _ = attn_map.shape

        #     # Select the first head for visualization
        #     attention_visualization = attn_map[0]  # First head

        #     # Plot and save the attention map
        #     output_file_path = "save_dir/darai/visualization/attention_map_baseline203050/attention_map_ep{}_{}_th.png".format(epoch, idx)
        #     plt.figure(figsize=(24, 12))
        #     plt.imshow(attention_visualization.detach().cpu().numpy(), cmap='hot', aspect='auto')
        #     plt.colorbar()
        #     plt.title("Activity: Example (Attention Map)")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Time Steps")
        #     plt.savefig(output_file_path)
        #     plt.close()
        ########################################################
        
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, epoch=0, idx=0):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, epoch=epoch, idx=idx)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
