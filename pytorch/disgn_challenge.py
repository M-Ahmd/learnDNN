import torch 
import torch.nn as nn

class Autoencode1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=60, out_features=40),
            nn.Linear(in_features=40, out_features=20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=20, out_features=40),
            nn.Linear(in_features=40, out_features=60)
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


class Autoencode2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feature=40, out_feature=15)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_feature=15, out_feature=40)
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=75),
            nn.Linear(in_features=75, out_features=25)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=25, out_features=75),
            nn.Linear(in_features=75, out_features=100)
        )
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


class AutoencoderMerger(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae_x1 = Autoencode1()
        self.ae_x2 = Autoencode2()
        self.ae_x = Autoencoder()

    def forward(self, x):
        x1 = self.ae_x1(x)
        x2 = self.ae_x2(x1)
        x = torch.hstack([x1, x2])

        x = x + self.ae_x(x)
        return x1, x2, x

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = AutoencoderMerger()
        self.regressor = nn.Sequential(
            nn.Linear(100, 50),
            torch.ReLU(),
            nn.Linear(50, 25),
            torch.ReLU(),
            nn.Linear(25, 1)
        )
    def forward(self, x):
        x1, x2 , x = self.autoencoder(x)
        regres = self.regressor(x)
        return x1, x2, x, regres

def drop_feature(x, drop_prob):
    drop_prob = torch.rand(1) * drop_prob

    mask = torch.rand_like(x) > drop_prob
    dropped_x = x * mask.float()
    return dropped_x


drop_prob = 0.9
criterion = nn.MSELoss()
regressor = RegressionModel()

x1 = torch.randn(32, 60)
x2 = torch.randn(32, 40)

x = torch.hstack([x1, x2])
gt = torch.randn(32, 1)

x1_noise = drop_feature(x1, drop_prob)
x2_noise = drop_feature(x2, drop_prob)

x1_rec, x2_rec, x_rec, pred = regressor(x1_noise, x2_noise)


loss1 = credits(x1, x1_rec)
loss2 = credits(x2, x2_rec)
loss3 = credits(x, x_rec)
loss4 = credits(gt, pred)
loss = (loss1 + loss2 + loss3 + loss4) / 4.0
