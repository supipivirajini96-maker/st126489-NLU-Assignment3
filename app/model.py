
import torch,  torchtext
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        return src
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len    = src.shape[1]

        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]

        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src: [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]

        return src
 
 # General Attention Mechanism
class GeneralAttention(nn.Module):
    """
    General Attention: ei = s^T * hi
    where s is the decoder state and hi is encoder hidden state
    Equation: ei = s^T * W * hi
    """
    def __init__(self, hid_dim, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.W = nn.Linear(hid_dim, hid_dim, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, query_len, hid_dim]
        # key: [batch_size, key_len, hid_dim]
        # value: [batch_size, value_len, hid_dim]

        # Transform key using W
        key_transformed = self.W(key)  # [batch_size, key_len, hid_dim]

        # Compute energy: query @ key_transformed^T
        energy = torch.bmm(query, key_transformed.transpose(1, 2)) / self.scale
        # [batch_size, query_len, key_len]

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to values
        context = torch.bmm(attention, value)
        # [batch_size, query_len, hid_dim]

        return context, attention
    

# Additive Attention Mechanism
class AdditiveAttention(nn.Module):
    """
    Additive Attention (Bahdanau): ei = v^T * tanh(W1*hi + W2*s)
    where v, W1, W2 are learnable parameters
    """
    def __init__(self, hid_dim, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.W1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W2 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, query_len, hid_dim]
        # key: [batch_size, key_len, hid_dim]
        # value: [batch_size, value_len, hid_dim]

        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        # Expand dimensions for broadcasting
        # query_expanded: [batch_size, query_len, 1, hid_dim]
        # key_expanded: [batch_size, 1, key_len, hid_dim]
        query_expanded = query.unsqueeze(2)
        key_expanded = key.unsqueeze(1)

        # Transform: W1*key + W2*query
        W1_key = self.W1(key_expanded)  # [batch_size, 1, key_len, hid_dim]
        W2_query = self.W2(query_expanded)  # [batch_size, query_len, 1, hid_dim]

        # Add and apply tanh activation
        energy = torch.tanh(W1_key + W2_query)  # [batch_size, query_len, key_len, hid_dim]

        # Apply v: [batch_size, query_len, key_len, 1]
        energy = self.v(energy).squeeze(-1)  # [batch_size, query_len, key_len]

        # Scale
        energy = energy / self.scale

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to values
        context = torch.bmm(attention, value)
        # [batch_size, query_len, hid_dim]

        return context, attention
    



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, attention_type='dot'):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.attention_type = attention_type

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Initialize attention mechanism
        if attention_type == 'general':
            self.attention = GeneralAttention(self.head_dim, device)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(self.head_dim, device)
        else:  # dot product (original)
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.attention_type in ['general', 'additive']:
            # For general and additive attention, process each head
            attention_outputs = []
            attention_weights = []

            for h in range(self.n_heads):
                q_h = Q[:, h, :, :]  # [batch_size, query_len, head_dim]
                k_h = K[:, h, :, :]  # [batch_size, key_len, head_dim]
                v_h = V[:, h, :, :]  # [batch_size, value_len, head_dim]

                # For mask, squeeze out the head dimension (middle dimension)
                # mask is [batch_size, 1, query_len/1, key_len]
                # We need to squeeze to [batch_size, query_len/1, key_len]
                if mask is not None:
                    mask_h = mask.squeeze(1)  # Remove the 1 dimension
                else:
                    mask_h = None

                out_h, att_h = self.attention(q_h, k_h, v_h, mask_h)
                attention_outputs.append(out_h)
                attention_weights.append(att_h)

            x = torch.stack(attention_outputs, dim=1)
            attention = torch.stack(attention_weights, dim=1)
        else:
            # Original dot product attention
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
            if mask is not None:
                energy = energy.masked_fill(mask == 0, -1e10)
            attention = torch.softmax(energy, dim=-1)
            x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]

        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        return trg, attention
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device,max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                            for _ in range(n_layers)])
        self.fc_out        = nn.Linear(hid_dim, output_dim)
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg: [batch_size, trg len, hid dim]
        #attention: [batch_size, n heads, trg len, src len]

        output = self.fc_out(trg)
        #output = [batch_size, trg len, output_dim]

        return output, attention
    

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention
    

