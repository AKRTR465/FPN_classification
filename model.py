from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


class FPN_classification(nn.Module):
    """
    FPN + 分类器的完整模型（不使用预训练backbone）
    """
    def __init__(self, 
                 in_channels: int = 3,
                 fpn_out_channels: int = 256,
                 num_classes: int = 5,
                 mlp_hidden_dims: list = [512, 512]):
        super(FPN_classification, self).__init__()
        
        # 直接使用自定义的FPN backbone（4层残差块）
        self.fpn_backbone = BackbonewithFPN(
            in_channels=in_channels,
            out_channels=fpn_out_channels
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP分类器
        self.classifier = self._mlp(fpn_out_channels, mlp_hidden_dims, num_classes)
        
    def _mlp(self, in_features, hidden_dims, num_classes):
        """
        构建MLP分类器
        """
        layers = []
        
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 获取FPN融合后的第一层特征图
        fpn_features = self.fpn_backbone(x)
        
        # 全局平均池化
        pooled_features = self.global_pool(fpn_features)
        pooled_features = pooled_features.flatten(1)
        
        # MLP分类
        output = self.classifier(pooled_features) 
        
        return output


class Bottleneck(nn.Module):
    """
    ResNet的bottleneck块
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # bottleneck结构: 1x1降维 -> 3x3卷积 -> 1x1升维
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接(如果输入输出通道数不同或stride!=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResidualStage(nn.Module):
    """
    残差阶段：包含多个Bottleneck块
    """
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super(ResidualStage, self).__init__()
        
        layers = []
        # 第一个block可能需要下采样
        layers.append(Bottleneck(in_channels, out_channels, stride=stride))
        
        # 后续blocks保持分辨率不变
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, stride=1))
        
        self.stage = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.stage(x)


class BackbonewithFPN(nn.Module):
    """
    4层残差块构成的backbone + FPN结构，输出第一层融合特征图
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 256):
        super().__init__()
        
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4个残差阶段，逐渐增加通道数和减小分辨率
        # Stage 1: 64 -> 256, stride=1 (分辨率不变)
        self.stage1 = ResidualStage(64, 256, num_blocks=3, stride=1)
        
        # Stage 2: 256 -> 512, stride=2 (分辨率减半)
        self.stage2 = ResidualStage(256, 512, num_blocks=4, stride=2)
        
        # Stage 3: 512 -> 1024, stride=2 (分辨率减半)
        self.stage3 = ResidualStage(512, 1024, num_blocks=6, stride=2)
        
        # Stage 4: 1024 -> 2048, stride=2 (分辨率减半)
        self.stage4 = ResidualStage(1024, 2048, num_blocks=3, stride=2)
        
        # FPN模块
        self.fpn = FPN_with_Bottleneck(out_channels=out_channels)
        self.out_channels = out_channels
    
    def forward(self, x):
        # 通过stem
        x = self.stem(x)
        
        # 通过4个残差阶段，收集特征
        features = OrderedDict()
        x = self.stage1(x)
        features['layer1'] = x
        
        x = self.stage2(x)
        features['layer2'] = x
        
        x = self.stage3(x)
        features['layer3'] = x
        
        x = self.stage4(x)
        features['layer4'] = x
        
        # FPN融合
        x = self.fpn(features)
        return x


class FPN_with_Bottleneck(nn.Module):
    """
    使用4个Bottleneck层的FPN结构，在融合时添加激活函数
    """
    
    def __init__(self, out_channels=256):
        super().__init__()
        
        in_channels_list = [256, 512, 1024, 2048]
        
        # 为每层创建1x1卷积调整通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1) 
            for in_c in in_channels_list
        ])
        
        # 4个bottleneck层用于特征融合
        self.bottleneck_layers = nn.ModuleList([
            Bottleneck(out_channels, out_channels, stride=1)
            for _ in range(4)
        ])
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: dict):
        # 获取4层特征
        features = list(x.values())
        
        # 通过lateral_convs调整通道数
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 从顶层(最小分辨率)开始，自顶向下融合
        results = []
        
        # 最后一层直接通过bottleneck
        last = self.bottleneck_layers[-1](laterals[-1])
        results.append(last)
        
        # 由后向前融合
        for idx in range(len(laterals)-2, -1, -1):
            # 上采样
            feat_shape = laterals[idx].shape[-2:]
            upsampled = F.interpolate(results[-1], size=feat_shape, mode='nearest')
            
            # 相加融合 + 激活函数
            fused = laterals[idx] + upsampled
            fused = self.bn(fused)
            fused = self.relu(fused)
            
            # 通过bottleneck层
            fused = self.bottleneck_layers[idx](fused)
            
            results.insert(0, fused)
        
        # 只返回第一层(最大分辨率)的融合特征图
        return results[0]

def fpn_classification(num_classes=5):
    return FPN_classification(
        in_channels=3,
        fpn_out_channels=256,
        num_classes=num_classes,
        mlp_hidden_dims=[512, 512]
    )

if __name__ == "__main__":
    model = FPN_classification(
        in_channels=3,
        fpn_out_channels=256,
        num_classes=5,
        mlp_hidden_dims=[512, 512]
    )
    
    print(model)
