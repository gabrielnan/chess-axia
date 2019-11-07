from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.loss_fn = nn.BCELoss()
        self.layers = [64 * 6 * 2 + 5 + 64, 600, 400, 200, 100]
        self.negative_slope = 1e-2
        self.encoder = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1]),
            nn.BatchNorm1d(self.layers[1]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[1], self.layers[2]),
            nn.BatchNorm1d(self.layers[2]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[2], self.layers[3]),
            nn.BatchNorm1d(self.layers[3]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[3], self.layers[4]),
            nn.BatchNorm1d(self.layers[4]),
            nn.LeakyReLU(self.negative_slope),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.layers[4], self.layers[3]),
            nn.BatchNorm1d(self.layers[3]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[3], self.layers[2]),
            nn.BatchNorm1d(self.layers[2]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[2], self.layers[1]),
            nn.BatchNorm1d(self.layers[1]),
            nn.LeakyReLU(self.negative_slope),

            nn.Linear(self.layers[1], self.layers[0]),
            nn.BatchNorm1d(self.layers[0]),
            nn.Sigmoid()
        )

    def forward(self, input):
        code = self.encoder(input)
        return self.decoder(code)

    def loss(self, input):
        return self.loss_fn(self.forward(input), input)

