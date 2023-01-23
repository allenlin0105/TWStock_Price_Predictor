from torch import nn

class PricePredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=args.hidden_size,
                            num_layers=args.n_layer,
                            dropout=args.dropout,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(args.hidden_size, 1, bias=False)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.fc(x)
        return x
