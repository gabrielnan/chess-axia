from torch import nn
from utils import INPUT_DIM, PIECES
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, dim=100, hidden=None):
        super(AutoEncoder, self).__init__()

        self.loss_fn = nn.BCELoss()
        if hidden is None:
            hidden = [600, 400, 200]
        hidden = [INPUT_DIM] + hidden + [dim]
        self.dim = dim
        self.negative_slope = 1e-2
        self.encoder = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[1], hidden[2]),
            nn.BatchNorm1d(hidden[2]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[2], hidden[3]),
            nn.BatchNorm1d(hidden[3]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[3], hidden[4]),
            nn.BatchNorm1d(hidden[4]),
            nn.LeakyReLU(self.negative_slope),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden[4], hidden[3]),
            nn.BatchNorm1d(hidden[3]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[3], hidden[2]),
            nn.BatchNorm1d(hidden[2]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[2], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(hidden[1], hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)

    def encode(self, x):
        return self.encoder(x)

    def loss(self, input):
        return self.loss_fn(self.forward(input), input)


class Valuator(nn.Module):
    def __init__(self, dim, hidden=None):
        super(Valuator, self).__init__()

        if hidden is None:
            hidden = [400, 200, 100]
        layers = [dim] + hidden + [1]
        negative_slope = 0.02

        self.model = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.BatchNorm1d(layers[1]),
            nn.LeakyReLU(negative_slope),

            nn.Linear(layers[1], layers[2]),
            nn.BatchNorm1d(layers[2]),
            nn.LeakyReLU(negative_slope),

            nn.Linear(layers[2], layers[3]),
            nn.BatchNorm1d(layers[3]),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class PieceValuator(nn.Module):
    def __init__(self, autoencoder, valuator=None):
        super(PieceValuator, self).__init__()
        self.autoencoder = autoencoder

        if valuator is None:
            valuator = self.Valuator(self.autoencoder.dim)
        self.valuator = valuator


    def forward(self, input):
        code = self.autoencoder.encode(input)
        return self.valuator(code)


class BoardValuator(nn.Module):
    def __init__(self, autoencoder, pawn=None, knight=None, bishop=None,
                 rook=None, queen=None, king=None):
        super(BoardValuator, self).__init__()

        self.models = {
            'p': PieceValuator(autoencoder) if pawn is None else pawn,
            'n': PieceValuator(autoencoder) if knight is None else knight,
            'b': PieceValuator(autoencoder) if bishop is None else bishop,
            'r': PieceValuator(autoencoder) if rook is None else rook,
            'q': PieceValuator(autoencoder) if queen is None else queen,
            'k': PieceValuator(autoencoder) if king is None else king,
        }
        self.loss_fn = nn.BCELoss()

    def forward(self, input):
        outs = []
        for color in range(2):
            out = 0
            for piece in PIECES:
                for x in input[color][piece]:
                    out = out + self.models[piece](x)
            outs.append(out)
        return F.sigmoid(outs[0] - outs[1])

    def loss(self, input, label):
        self.loss_fn(self.forward(input), label)

class Comparator(nn.Module):
    def __init__(self, valuator1, valuator2):
        super(Comparator, self).__init__()
        self.valuator1 = valuator1
        self.valuator2 = valuator2

    def forward(self, input1, input2):
        out1 = F.sigmoid(self.valuator1(input1))
        out2 = F.sigmoid(self.valuator2(input2))
        raise NotImplementedError

        # todo: stack and return



