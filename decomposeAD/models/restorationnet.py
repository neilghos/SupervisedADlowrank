import torch
import torch.nn as nn

try:
    from models.encoder import Encoder
    from models.decompose_net import Decomposenet
    from models.fuser import FusionDecoder
except ImportError:
    from encoder import Encoder
    from decompose_net import Decomposenet
    from fuser import FusionDecoder

class AnalyticalRestorationNet(nn.Module):
    def __init__(self, in_channels=3, num_decomp_maps=8):
        super().__init__()
        
        # 1. The Shrinker (Encoder)
        self.backbone = Encoder(in_channels=in_channels)
        
        # 2. The Map Predictor
        self.decomp_head = Decomposenet(out_channels=num_decomp_maps)
        

        self.fusion_decoder = FusionDecoder(decomp_channels=num_decomp_maps, out_channels=in_channels)
        
    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        pred_maps = self.decomp_head(feat1, feat2, feat3)
        restored_rgb = self.fusion_decoder(feat1, feat2, feat3, pred_maps)
        return pred_maps, restored_rgb
