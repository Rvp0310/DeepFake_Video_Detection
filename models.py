import torch
import torch.nn as nn
import torchvision.models as models

class Model0(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_net = models.mobilenet_v2(weights="IMAGENET1K_V1").features

        for param in self.spatial_net.parameters():
            param.requires_grad = True
            
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.freq_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.attn = nn.Linear(1280 + 16, 1)
        self.fc = nn.Sequential(
            nn.Linear(1296, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        

    def forward(self, spatial, frequency):
        B, T, C, H, W = spatial.shape

        spatial = spatial.view(B * T, C, H, W)
        frequency = frequency.view(B * T, 1, H, W)

        s_feat = self.spatial_net(spatial)
        s_feat = self.pool(s_feat)
        f_feat = self.freq_net(frequency)

        s_feat = s_feat.view(B, T, -1)
        f_feat = f_feat.view(B, T, -1)

        combined = torch.cat([s_feat, f_feat], dim = 2)

        weights = torch.softmax(self.attn(combined), dim = 1)
        
        combined = (weights * combined).sum(dim = 1)
        
        out = self.fc(combined)

        return out

class Model1(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_net = models.mobilenet_v2(weights="IMAGENET1K_V1").features

        for param in self.spatial_net[:-20].parameters():
            param.requires_grad = False

        for param in self.spatial_net[-20:].parameters():
            param.requires_grad = True
            
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.freq_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.attn = nn.Sequential(
            nn.Linear(1312, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.norm = nn.LayerNorm(1312)
        
        self.fc = nn.Sequential(
            nn.Linear(1312, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 1)
        )

    def forward(self, spatial, frequency):
        B, T, C, H, W = spatial.shape

        spatial = spatial.view(B * T, C, H, W)
        frequency = frequency.view(B * T, 1, H, W)

        s_feat = self.spatial_net(spatial)
        s_feat = self.pool(s_feat)
        f_feat = self.freq_net(frequency)

        s_feat = s_feat.view(B, T, -1)
        f_feat = f_feat.view(B, T, -1)

        combined = torch.cat([s_feat, f_feat], dim = 2)
        combined = self.norm(combined)

        weights = torch.softmax(self.attn(combined), dim = 1)
        
        combined = (weights * combined).sum(dim = 1)
        
        out = self.fc(combined)

        return out
    

class Model2(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_net = models.mobilenet_v2(weights="IMAGENET1K_V1").features

        for param in self.spatial_net[-4:].parameters():
            param.requires_grad = False
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.freq_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.temporal = nn.GRU(1312, 256, batch_first=True)

        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.norm = nn.LayerNorm(1312)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 1)
        )

    def forward(self, spatial, frequency):
        B, T, C, H, W = spatial.shape

        spatial = spatial.view(B * T, C, H, W)
        frequency = frequency.view(B * T, 1, H, W)

        s_feat = self.spatial_net(spatial)
        s_feat = self.pool(s_feat)
        f_feat = self.freq_net(frequency)

        s_feat = s_feat.view(B, T, -1)
        f_feat = f_feat.view(B, T, -1)

        combined = torch.cat([s_feat, f_feat], dim = 2)
        combined = self.norm(combined)

        combined, _ = self.temporal(combined)

        weights = torch.softmax(self.attn(combined), dim = 1)
        
        combined = (weights * combined).sum(dim = 1)
        
        out = self.fc(combined)

        return out