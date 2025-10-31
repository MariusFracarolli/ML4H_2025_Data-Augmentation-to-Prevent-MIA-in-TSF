import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.dec = nn.TransformerDecoderLayer(d_model=configs.dec_in, dim_feedforward=configs.dec_in, nhead=1, batch_first=True)
        self.linearize = nn.Linear(configs.d_model, configs.dec_in)

        self.dec2 = nn.TransformerDecoderLayer(d_model=configs.dec_in, dim_feedforward=configs.dec_in, nhead=1, batch_first=True)
        self.linearize2 = nn.Linear(configs.dec_in, 1)
        self.dec_in = configs.dec_in
        
    def forward(self, emb, x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, 
                trainn=False, tgt=None, backprop=False, mode="Forecasting"):
        dec_in = self.dec_in 
        enc_out, attns = self.encoder(emb, attn_mask=enc_self_mask)
        var_y = enc_out # transformer embedding
        if trainn:
            mask_new = torch.nn.Transformer.generate_square_subsequent_mask(25).cuda()
            linear_memory = self.linearize(enc_out)
            start_input = torch.zeros(enc_out.size(0), 1, dec_in).cuda()
            tgt = torch.concat([start_input, tgt], dim=1)
            result=self.dec(tgt, linear_memory, mask_new)
            return result[:, 1:, :]           
        else:
            if mode == "Classification":
                output = torch.zeros(enc_out.size(0), 1, dec_in).cuda()
                linear_memory  = self.linearize(enc_out).detach() if not backprop else self.linearize(enc_out)
                res = self.dec2(output.detach() if not backprop else output, linear_memory)
                res = self.linearize2(res)
                return torch.sigmoid(res)
            output = torch.zeros(enc_out.size(0), 1, dec_in).cuda()
            linear_memory  = self.linearize(enc_out)
            for i in range(24):
                res = self.dec(output.detach() if not backprop else output, linear_memory)
                output = torch.concat([output, res[:,-1:,:]], dim=1)
            return output[:, 1:, :], emb, var_y 
