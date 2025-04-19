import torch.nn as nn

class SOHPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # 解码器部分，用于重构输入
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        # 残差链接
        self.skip = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded + self.skip(x)  # 残差连接