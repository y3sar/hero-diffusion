from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(OrderedDict([
            ('l1', nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=0)),
            ('relu', nn.ReLU()),
            ('l2', nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=0)),
            ('relu', nn.ReLU())
        ]))


    def forward(self, x):
        
        return self.conv_block(x)
    

class BasicUnet(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([ConvBlock(channels[ch], channels[ch+1]) for ch in range(len(channels)-1)])
        # self.decoder_blocks = nn.ModuleList([ConvBlock(channels[ch], channels[ch-1]) for ch in range(len(channels)-1, 0, -1)])

        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for ch in range(len(channels)-1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(channels[ch], channels[ch]//2, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(channels[ch], channels[ch-1]))

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.activations = []

    def contract_network(self, x):
        
        for block in self.encoder_blocks[:-1]:
            x = block(x)
            self.activations.append(x)

            x = self.maxpool(x)

        x = self.encoder_blocks[-1](x)

        return x
        

    def upsampling_network(self, x):
        
        for idx in range(0, len(self.decoder_blocks)-1):
            upsample = self.up_convs[idx]
            conv_block = self.decoder_blocks[idx]

            x = upsample(x)
            
            enc_activation = self.crop(self.activations[-(idx+1)], x)

            x  = torch.cat([x, enc_activation], dim=1)

            x = conv_block(x)

        x = self.decoder_blocks[-1](x)

        return x


    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    

    def forward(self, x):

        # Contract steps
        out = unet.contract_network(x)

        # Upsampling steps
        out = unet.upsampling_network(out)

        return out

        


        


if __name__ == '__main__':

    x = torch.randn(1, 3, 572, 572)

    # Unet convblock with repeating convolutions
    block = ConvBlock(3, 64)
    y = block(x)

    print("Conv Block output: ", y.shape)

    unet = BasicUnet()

    print("Unet Output: ", unet(x).shape)
