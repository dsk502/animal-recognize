import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSelfAttention(nn.Module):
    """改进的自注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        
        attn = self.softmax(torch.bmm(q, k))
        attn = self.attn_drop(attn)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return self.gamma * out.view(B, C, H, W) + x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedResidualBlock(nn.Module):
    """增强的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)
        
        self.dropout = nn.Dropout2d(0.1) if stride == 1 else None

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        out += self.shortcut(x)
        return self.activation(out)

class StochasticDepth(nn.Module):
    """随机深度模块"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or torch.rand(1)[0] > self.p:
            return x
        return torch.zeros_like(x)

class EnhancedAnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 输入: 3x128x128
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2)  # 64x64x64
        )
        
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(EnhancedResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(EnhancedResidualBlock(out_channels, out_channels))
            layers.append(EnhancedSelfAttention(out_channels))
            layers.append(StochasticDepth(0.1))
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 修改处：将'silu'替换为'relu'以兼容初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        return self.classifier(x)