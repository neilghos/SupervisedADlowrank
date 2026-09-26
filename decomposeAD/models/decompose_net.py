import torch
import torch.nn as nn


class Decomposenet(nn.Module):
    def __init__(self, out_channels = 8):
        super().__init__()
        
        self.up1 = nn.ConvTranspose2d(256,128, kernel_size= 2, stride = 2) #for feat3, converts 64->128 map size and channel 256->128
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)

        )
        #conv1 eats the (up1(feat3)+feat2)) and blows back the channel nums to 128
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2) #for up1(feat3)+feat2, converts 128->256 map size and channel 128->64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
            )
        
        #conv2 takes the final up2(conv1)+feat1 and blows back the channel numes to 64 and then to 8.


        
        
        
        #format = (channel, length, width) of output feature map
        #feat1 = (64, 256, 256)
        #feat2 =(128,128,128)
        #feat3 = (256, 64, 64)
    def forward(self, feat1, feat2, feat3):
        
        x = self.up1(feat3) #feat 3 gets upscaled from 256, 64,64 to 128,128,128
        x = torch.cat([x,feat2], dim = 1) #(256,128,128) feat2 was already 128 128 128 so add it to x which is 128 128 128
        
        x = self.conv1(x) #128 128 128
        
        x = self.up2(x) #64, 256, 256
        
        x = torch.cat([x, feat1], dim = 1) #128, 256,256
        
        maps = self.conv2(x) #8, 256, 256
        
        return maps
    
    #mapsize = 8,256,256