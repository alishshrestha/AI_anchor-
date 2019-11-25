import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()

        bn = None
        if batch_size == 1:
            bn = False # Instance Normalization
        else:
            bn = True # Batch Normalization

        # [3x256x256] -> [64x128x128]
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        # -> [128x64x64]
        conv2 = [nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1)]
        if bn == True:
            conv2 += [nn.BatchNorm2d(128)]
        else:
            conv2 += [nn.InstanceNorm2d(128)]
        self.conv2 = nn.Sequential(*conv2)

        # -> [256x32x32]
        conv3 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(128, 256, 4, 2, 1)]
        if bn == True:
            conv3 += [nn.BatchNorm2d(256)]
        else:
            conv3 += [nn.InstanceNorm2d(256)]
        self.conv3 = nn.Sequential(*conv3)

        # -> [512x16x16]
        conv4 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(256, 512, 4, 2, 1)]
        if bn == True:
            conv4 += [nn.BatchNorm2d(512)]
        else:
            conv4 += [nn.InstanceNorm2d(512)]
        self.conv4 = nn.Sequential(*conv4)

        # -> [512x8x8]
        conv5 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 512, 4, 2, 1)]
        if bn == True:
            conv5 += [nn.BatchNorm2d(512)]
        else:
            conv5 += [nn.InstanceNorm2d(512)]
        self.conv5 = nn.Sequential(*conv5)

        # -> [512x4x4]
        conv6 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 512, 4, 2, 1)]
        if bn == True:
            conv6 += [nn.BatchNorm2d(512)]
        else:
            conv6 += [nn.InstanceNorm2d(512)]
        self.conv6 = nn.Sequential(*conv6)

        # -> [512x2x2]
        conv7 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 512, 4, 2, 1)]
        if bn == True:
            conv7 += [nn.BatchNorm2d(512)]
        else:
            conv7 += [nn.InstanceNorm2d(512)]
        self.conv7 = nn.Sequential(*conv7)

        # -> [512x1x1]
        conv8 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 512, 4, 2, 1)]
        if bn == True:
            conv8 += [nn.BatchNorm2d(512)]
        else:
            conv8 += [nn.InstanceNorm2d(512)]
        self.conv8 = nn.Sequential(*conv8)

        # -> [512x2x2]
        deconv8 = [nn.ReLU(),
                   nn.ConvTranspose2d(512, 512, 4, 2, 1)]
        if bn == True:
            deconv8 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]
        else:
            deconv8 += [nn.InstanceNorm2d(512), nn.Dropout(0.5)]
        self.deconv8 = nn.Sequential(*deconv8)

        # [(512+512)x2x2] -> [512x4x4]
        deconv7 = [nn.ReLU(),
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]
        if bn == True:
            deconv7 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]
        else:
            deconv7 += [nn.InstanceNorm2d(512), nn.Dropout(0.5)]
        self.deconv7 = nn.Sequential(*deconv7)

        # [(512+512)x4x4] -> [512x8x8]
        deconv6 = [nn.ReLU(),
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]
        if bn == True:
            deconv6 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]
        else:
            deconv6 += [nn.InstanceNorm2d(512), nn.Dropout(0.5)]
        self.deconv6 = nn.Sequential(*deconv6)

        # [(512+512)x8x8] -> [512x16x16]
        deconv5 = [nn.ReLU(),
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]
        if bn == True:
            deconv5 += [nn.BatchNorm2d(512)]
        else:
            deconv5 += [nn.InstanceNorm2d(512)]
        self.deconv5 = nn.Sequential(*deconv5)

        # [(512+512)x16x16] -> [256x32x32]
        deconv4 = [nn.ReLU(),
                   nn.ConvTranspose2d(512 * 2, 256, 4, 2, 1)]
        if bn == True:
            deconv4 += [nn.BatchNorm2d(256)]
        else:
            deconv4 += [nn.InstanceNorm2d(256)]
        self.deconv4 = nn.Sequential(*deconv4)

        # [(256+256)x32x32] -> [128x64x64]
        deconv3 = [nn.ReLU(),
                   nn.ConvTranspose2d(256 * 2, 128, 4, 2, 1)]
        if bn == True:
            deconv3 += [nn.BatchNorm2d(128)]
        else:
            deconv3 += [nn.InstanceNorm2d(128)]
        self.deconv3 = nn.Sequential(*deconv3)

        # [(128+128)x64x64] -> [64x128x128]
        deconv2 = [nn.ReLU(),
                   nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1)]
        if bn == True:
            deconv2 += [nn.BatchNorm2d(64)]
        else:
            deconv2 += [nn.InstanceNorm2d(64)]
        self.deconv2 = nn.Sequential(*deconv2)

        # [(64+64)x128x128] -> [3x256x256]
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, x):

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)

        d7 = self.deconv8(c8)
        d7 = torch.cat((c7, d7), dim=1)
        d6 = self.deconv7(d7)
        d6 = torch.cat((c6, d6), dim=1)
        d5 = self.deconv6(d6)
        d5 = torch.cat((c5, d5), dim=1)
        d4 = self.deconv5(d5)
        d4 = torch.cat((c4, d4), dim=1)
        d3 = self.deconv4(d4)
        d3 = torch.cat((c3, d3), dim=1)
        d2 = self.deconv3(d3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1)
        out = self.deconv1(d1)

        return out

class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()

        bn = None
        if batch_size == 1:
            bn = False  # Instance Normalization
        else:
            bn = True  # Batch Normalization

        # [(3+3)x256x256] -> [64x128x128] -> [128x64x64]
        main = [nn.Conv2d(3*2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(128)]
        else:
            main += [nn.InstanceNorm2d(128)]

        # -> [256x32x32]
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(128, 256, 4, 2, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(256)]
        else:
            main += [nn.InstanceNorm2d(256)]

        # -> [512x31x31] (Fully Convolutional)
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(256, 512, 4, 1, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(512)]
        else:
            main += [nn.InstanceNorm2d(512)]

        # -> [1x30x30] (Fully Convolutional, PatchGAN)
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(512, 1, 4, 1, 1),
                  nn.Sigmoid()]

        self.main = nn.Sequential(*main)

    def forward(self, x1, x2): # One for Real, One for Fake
        out = torch.cat((x1, x2), dim=1)
        return self.main(out)