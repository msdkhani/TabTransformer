import torch
import torch.nn as nn
import math
import yaml

def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Load configuration from YAML file
config = load_config('config.yaml')
SEQUENCE_LENGTH = config['SEQUENCE_LENGTH']
VOCAB_SIZE = config['VOCAB_SIZE']
EMBED_DIM = config['EMBED_DIM']
NUM_LAYERS = config['NUM_LAYERS']
NUM_HEADS = config['NUM_HEADS']
MAX_LEN = config['MAX_LEN']
DROPOUT = config['DROPOUT']

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.2):
        """
        Initialize PositionalEncoding module.
        Args:
            max_len (int): Maximum sequence length.
            d_model (int): Dimensionality of the model.
            dropout (float): Dropout probability.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of PositionalEncoding module.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_len=100, dropout=0.2):
        """
        Initialize Encoder module.
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the embedding.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=max_len, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of Encoder module.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor.
        """
        emb = self.emb(x)
        x = self.pos_encoder(emb)
        x = self.encoder(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_len=100, dropout=0.2):
        """
        Initialize Decoder module.
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the embedding.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()  # Call the superclass's __init__() method

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(max_len=max_len, d_model=embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate square subsequent mask.
        Args:
            sz (int): Size of the mask.
        Returns:
            Tensor: Square subsequent mask.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        Forward pass of Decoder module.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor.
        """
        emb = self.emb(x)
        emb = self.pos_encoder(emb)  # Positional encoding added after embedding
        input_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.decoder(emb, memory=emb, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        """
        Initialize Transformer module.
        Args:
            config (dict): Configuration dictionary.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size=config['VOCAB_SIZE'],
            embed_dim=config['EMBED_DIM'],
            num_layers=config['NUM_LAYERS'],
            num_heads=config['NUM_HEADS'],
            max_len=config['MAX_LEN'],
            dropout=config['DROPOUT']
        )
        self.decoder = Decoder(
            vocab_size=config['VOCAB_SIZE'],
            embed_dim=config['EMBED_DIM'],
            num_layers=config['NUM_LAYERS'],
            num_heads=config['NUM_HEADS'],
            max_len=config['MAX_LEN'],
            dropout=config['DROPOUT']
        )

    def forward(self, src, tgt):
        """
        Forward pass of Transformer module.
        Args:
            src (Tensor): Source input tensor.
            tgt (Tensor): Target input tensor.
        Returns:
            Tensor: Output tensor.
        """
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output