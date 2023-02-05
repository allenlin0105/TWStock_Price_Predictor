import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fcs = nn.ModuleList()
        if n_layer == 1:
            self.fcs.append(nn.Linear(input_dim, output_dim))
        else:
            self.fcs.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(n_layer - 2):
                self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
            self.fcs.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x

class LSTMPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=args.hidden_size,
                            num_layers=args.n_layer,
                            dropout=args.dropout,
                            bidirectional=True,
                            batch_first=True)
        self.fc = FeedForward(args.fc_layer, args.hidden_size, args.fc_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    # Reference from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_layer = nn.Linear(1, args.d_model)
        self.positional_encoder = PositionalEncoding(
            d_model=args.d_model,
            dropout=args.dropout
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.d_model, 
                nhead=args.n_head,
                dim_feedforward=args.d_model,
                dropout=args.dropout
            ),
            num_layers=args.n_layer
        )
        self.fc = FeedForward(args.fc_layer, args.d_model * args.n_input_days, args.fc_dim, 1)
    
    def forward(self, x):  # x: [batch_size, seq_len, 1]
        x = self.input_layer(x)
        x = x.permute(1, 0, 2)  # x: [seq_len, batch_size, dim]
        x = self.positional_encoder(x)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # x: [batch_size, seq_len, dim]
        x = x.reshape(x.size(0), -1)  # x: [batch_size, seq_len * dim] 
        x = self.fc(x)
        return x


class TransformerPredictor(nn.Module):
    def __init__(self, args): 
        super().__init__()

        d_model = args.d_model
        n_head = args.n_head
        n_layer = args.n_layer
        dropout = args.dropout
        
        self.encoder_input_layer = nn.Linear(1, d_model)
        self.positional_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=dropout
            ),
            num_layers=n_layer
        )

        self.decoder_input_layer = nn.Linear(1, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=dropout
            ),
            num_layers=n_layer
        )
        self.decoder_fc = FeedForward(args.fc_layer, d_model, args.fc_dim, 1)

    def forward(self, x):
        source = x  # [batch_size, seq_len, 1]
        target = x[:, -1, :].unsqueeze(1)  # [batch_size, 1, 1]

        source = self.encoder_input_layer(source)
        source = source.permute(1, 0, 2)
        source = self.positional_encoder(source)
        source = self.encoder(source)

        decoder_output = self.decoder_input_layer(target)
        decoder_output = decoder_output.permute(1, 0, 2)
        decoder_output = self.decoder(decoder_output, source)
        decoder_output = decoder_output.permute(1, 0, 2)
        decoder_output = self.decoder_fc(decoder_output)
        decoder_output = decoder_output.squeeze(2)

        return decoder_output