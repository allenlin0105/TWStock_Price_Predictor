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
        self.fcs = nn.ModuleList()
        for _ in range(args.fc_layer - 1):
            self.fcs.append(nn.Linear(args.hidden_size, args.hidden_size))
        self.fcs.append(nn.Linear(args.hidden_size, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = h[-1]
        for fc in self.fcs:
            x = fc(x)
        return x
