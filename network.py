import torch
from torch import nn



class SoccerMap(nn.Module):
    def __init__(self):
        super(SoccerMap, self).__init__()
        self.FeaE = nn.Sequential(
            nn.Conv2d(16, 48, 5, padding=2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(48, 64, 5, padding=2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 1, 1)
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1, padding_mode='replicate')
        )
        self.maxPooling = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fusion = nn.Conv2d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):               # batch_size * C * W * H
        x_1 = x
        x_2 = self.maxPooling(x_1)
        x_3 = self.maxPooling(x_2)

        pred_y_1 = self.FeaE(x_1)
        pred_y_2 = self.FeaE(x_2)
        pred_y_3 = self.FeaE(x_3)

        y_3 = self.smooth(self.upSample(pred_y_3))
        y_2 = self.smooth(self.upSample(self.fusion(torch.cat((pred_y_2, y_3), 1))))
        y_1 = self.fusion(torch.cat((pred_y_1, y_2), 1))

        y = self.sigmoid(y_1)
        return y*0.5-1          # batch_size * 1 * W * H

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output_matrix, destination, label):
        output_matrix = torch.clamp(output_matrix, 0.00001, 0.99999)
        loss = 0.
        for i in range(len(output_matrix)):
            om = output_matrix[i]
            dn = destination[i].item()
            lb = 1  # 每个传球点都视为传球选择
            l = torch.pow((om.view(1,-1)[0][dn]-lb), 2)
            loss += l
        return loss/len(output_matrix)