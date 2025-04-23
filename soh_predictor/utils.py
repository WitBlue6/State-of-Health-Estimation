import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=2, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.fc(attn_output)

class SOHPredictor2(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            #nn.LayerNorm(256),  # 添加归一化
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            #nn.LayerNorm(64),
            nn.GELU()
        )
        self.attention = SelfAttention(64, num_heads=1, dropout=0.3)
        self.noise_filter = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # attn_output = self.attention(encoded.unsqueeze(1))  # 增加维度适配多头自注意力
        # # 噪声过滤器
        # filtered = self.noise_filter(attn_output.squeeze(1))  # 还原维度
        filtered = self.noise_filter(encoded)
        return self.decoder(filtered)
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        # 添加特征相似性约束
        cos_loss = 1 - nn.CosineSimilarity()(pred, target).mean()
        return self.alpha * mse_loss + (1-self.alpha) * cos_loss

class SOHPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            #nn.LayerNorm(256),  # 添加归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
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
    
class HubberLoss(nn.Module):
    """对微小误差不敏感的新型损失"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, true):
        diff = pred - true
        return torch.where(
            torch.abs(diff) < self.delta,
            0.5 * diff.pow(2),
            self.delta * (torch.abs(diff) - 0.5 * self.delta)
        ).mean()

