# model.py
import torch
import torch.nn as nn

class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True):
        super(EncoderBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x), x

class Encoder3D(nn.Module):
    def __init__(self, in_channels, base_filters=64, depth=4):
        super(Encoder3D, self).__init__()
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(depth):
            out_channels = base_filters * (2**i)
            encoder = EncoderBlock3D(current_channels, out_channels)
            self.encoders.append(encoder)
            current_channels = out_channels
        
        self.bottleneck_channels = current_channels

    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            features.append(skip)
        return x, features

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DecoderBlock3D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class MidLevelFusion(nn.Module):
    def __init__(self, channels, fusion_type='attention'):
        super(MidLevelFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Conv3d(channels * 2, channels, kernel_size=1)
        elif fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Conv3d(channels * 2, channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.fusion = nn.Conv3d(channels * 2, channels, kernel_size=1)
    
    def forward(self, s1_features, s2_features):
        if self.fusion_type == 'concat':
            combined = torch.cat([s1_features, s2_features], dim=1)
            return self.fusion(combined)
        elif self.fusion_type == 'attention':
            combined = torch.cat([s1_features, s2_features], dim=1)
            attention_weights = self.attention(combined)
            weighted_features = torch.cat([
                s1_features * attention_weights,
                s2_features * (1 - attention_weights)
            ], dim=1)
            return self.fusion(weighted_features)

class DualStreamUNET3D(nn.Module):
    def __init__(self, config):
        super(DualStreamUNET3D, self).__init__()
        self.s1_channels = len(config.get('s1_variables', ['VH', 'VV', 'incidence_angle', 'DEM']))
        self.s2_channels = len(config.get('s2_variables', ['B02', 'B03', 'B04', 'B08', 'B11', 
                                                          'NDVI', 'NDWI_veg', 'NDWI_water', 'DEM']))
        base_filters = config.get('base_filters', 64)
        depth = config.get('depth', 4)
        
        # Separate encoders for S1 and S2
        self.s1_encoder = Encoder3D(self.s1_channels, base_filters, depth)
        self.s2_encoder = Encoder3D(self.s2_channels, base_filters, depth)
        
        # Mid-level fusion module
        bottleneck_channels = base_filters * (2**(depth-1))
        self.fusion = MidLevelFusion(bottleneck_channels, fusion_type=config.get('fusion_type', 'attention'))
        
        # Decoder with skip connections
        self.decoders = nn.ModuleList()
        for i in range(depth-1, -1, -1):
            in_channels = base_filters * (2**i) * 2  # *2 for skip connections
            out_channels = base_filters * (2**max(0, i-1))
            self.decoders.append(DecoderBlock3D(in_channels, out_channels))
        
        # Final classification layer
        self.final_conv = nn.Conv3d(base_filters, config.get('num_classes', 2), kernel_size=1)

    def load_pretrained_encoders(self, s1_path, s2_path):
        s1_state = torch.load(s1_path)
        s2_state = torch.load(s2_path)
        self.s1_encoder.load_state_dict(s1_state['encoder_state_dict'])
        self.s2_encoder.load_state_dict(s2_state['encoder_state_dict'])
    
    def forward(self, s1, s2):
        # Get encoded features from both streams
        s1_bottleneck, s1_features = self.s1_encoder(s1)
        s2_bottleneck, s2_features = self.s2_encoder(s2)
        
        # Fuse the bottleneck features
        fused_features = self.fusion(s1_bottleneck, s2_bottleneck)
        
        # Decode with skip connections
        x = fused_features
        for i, decoder in enumerate(self.decoders):
            skip = torch.cat([s1_features[-(i+1)], s2_features[-(i+1)]], dim=1)
            x = decoder(x, skip)
        
        return self.final_conv(x)