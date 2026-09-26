import torch
import torch.nn as nn


class FusionDecoder(nn.Module):

    def __init__(self, decomp_channels=8, out_channels=3):
        super().__init__()
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 64 + decomp_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            
            nn.Sigmoid() 
        )
        #format = (channel, length, width) of output feature map
        #feat1 = (64, 256, 256)
        #feat2 =(128,128,128)
        #feat3 = (256, 64, 64)
        #maps = 8,256,256
    def forward(self, feat1, feat2, feat3, predicted_maps):
        x = self.up1(feat3) #feat3 gets upscalled to 128, 128,128
        x = torch.cat([x, feat2], dim=1) #256,128,128
        x = self.conv1(x) #128,128,128
        
        x = self.up2(x)  #64, 256,256
        
        x = torch.cat([x, feat1, predicted_maps], dim=1) #136, 256, 256
    
        rgb_restored = self.conv2(x) #3,256,256
        return rgb_restored #final regb image



