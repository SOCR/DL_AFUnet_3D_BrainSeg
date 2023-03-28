import torch.nn as nn
import torch
from torch.nn.modules import upsampling
import torchvision.transforms.functional as F


class MHSA(nn.Module):
    """
    Self-Attention.
    """
    def __init__(self, channels: int, dimension=2):
        super().__init__()
        if dimension == 2:
          conv = nn.Conv2d
          batchnorm = nn.BatchNorm2d
        elif dimension == 3:
          conv = nn.Conv3d
          batchnorm = nn.BatchNorm3d
        self.channels = channels
        self.filter_x = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
        )
        self.filter_g = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.filter_1 = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
            nn.Sigmoid()
        )
        # self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1):
        x = self.filter_x(x1)
        g = self.filter_g(x1)
        out = x+g
        out = self.relu(out)
        out = self.filter_1(out)
        # out = self.upsampling(out)
        out = x1*out

        return out


class Attn(nn.Module):
    """
    In this module we are going to convert x to x_hat which use the information of x and g.
    x: (N, C, H, W)
    g: (N, 2*C，H//2， W//2)
    x_hat: (N, C, H, W)
    """
    def __init__(self, channels: int, dimension=2):
        super().__init__()
        if dimension == 2:
          conv = nn.Conv2d
          batchnorm = nn.BatchNorm2d
          upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        elif dimension == 3:
          conv = nn.Conv3d
          batchnorm = nn.BatchNorm3d
          upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.channels = channels
        self.filter_x = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels)
        )
        self.filter_g = nn.Sequential(
            conv(2*channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.filter_1 = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
            nn.Sigmoid()
        )
        self.upsampling = upsample

    def forward(self, x1, g):
        x = self.filter_x(x1)
        g = self.filter_g(g)
        g = self.upsampling(g)
        out = x+g
        out = self.relu(out)
        out = self.filter_1(out)
        out = x1*out

        return out


class Attn_mul(nn.Module):
    """
    In this module we are going to convert x to x_hat which use the information of x and g.
    x: (N, C, H, W)
    g: (N, 2*C，H//2， W//2)
    x_hat: (N, C, H, W)
    """
    def __init__(self, channels: int, dimension=2):
        super().__init__()
        if dimension == 2:
          conv = nn.Conv2d
          batchnorm = nn.BatchNorm2d
          upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        elif dimension == 3:
          conv = nn.Conv3d
          batchnorm = nn.BatchNorm3d
          upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.channels = channels
        
        self.filter_x = nn.Sequential(
            conv(channels, channels, kernel_size=3, stride=2, padding=1),
            batchnorm(num_features=channels),
            nn.ReLU(inplace=True)
        )
        self.filter_g = nn.Sequential(
            conv(2*channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.filter_1 = nn.Sequential(
            conv(channels, channels, kernel_size=1, stride=1, padding=0),
            batchnorm(num_features=channels),
            nn.Sigmoid()
        )
        self.upsampling = upsample

    def forward(self, x1, g):
        x = self.filter_x(x1)
        g = self.filter_g(g)
        
        out = self.filter_1(out)
        out = self.upsampling(out)
        out = x1*out

        return out


class DoubleConvLayer(nn.Module):
    """
    Double Convolution Layer with Batchnorm and padding for convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2):
        super().__init__()
        if dimension == 2:
          conv = nn.Conv2d
          batchnorm = nn.BatchNorm2d
        elif dimension == 3:
          conv = nn.Conv3d
          batchnorm = nn.BatchNorm3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequential = nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            batchnorm(num_features=out_channels),
            nn.ReLU(inplace=True),
            conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            batchnorm(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.sequential(x)   


class AFUNet(nn.Module):
    """
    An example of f_1.
    Claim the dimension 2 as 2d or 3 as 3d.
    mhsa is the parameter controlling whether to use self-attention at the bottlenecks or not.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2, attn='yes', mhsa='none', dropout='none'):
        """
        
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d
        
        self.in_channels = in_channels
        self.mhsa = mhsa
        self.attn = attn
        self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)

        self.double_conv2 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)
        self.double_conv3 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)

        self.double_conv4 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)

        self.double_conv5 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)
        self.double_conv6 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)

        self.double_conv7 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)

        self.double_conv8 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv9 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)
        self.double_conv10 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv11 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)

        self.max_pool = max_pool
        self.out_activation = nn.Sigmoid()

        
        self.deconv1 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv2 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv3 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.deconv4 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv5 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.final_conv = conv(in_channels=16, out_channels=out_channels, kernel_size=1)

        if self.attn == 'yes':
          self.attn4 = Attn(32, dimension=dimension)
          self.attn2 = Attn(32, dimension=dimension)
          self.attn1 = Attn(16, dimension=dimension)

          self.attn3 = Attn(64, dimension=dimension)
          self.attn5 = Attn(64, dimension=dimension)

        if self.mhsa == 'yes':
          self.mhsa1 = MHSA(128, dimension=dimension)
          self.mhsa2 = MHSA(128, dimension=dimension)
        
    def forward(self, x):
        
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)

        x = self.double_conv3(x)
        identity3 = x
        x = self.max_pool(x)

        x = self.double_conv8(x)
        if self.mhsa == 'yes':
          x = self.mhsa1(x)

        if self.attn == 'yes':
          identity3 = self.attn3(identity3, x)
        x = self.deconv4(x)
        # identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv9(x)
        
        if self.attn == 'yes':
          identity2 = self.attn2(identity2, x)
        x = self.deconv1(x)
        # identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv4(x)
        identity4 = x
        x = self.max_pool(x)

        x = self.double_conv5(x)
        identity5 = x
        x = self.max_pool(x)

        x = self.double_conv10(x)
        if self.mhsa == 'yes':
          x = self.mhsa2(x)

        if self.attn == 'yes':
          identity5 = self.attn3(identity5, x)
        x = self.deconv5(x)
        # identity5 = F.resize(identity5, size=x.shape[2:])
        x = torch.cat((x, identity5), dim=1)
        x = self.double_conv11(x)
        
        if self.attn == 'yes':
          identity4 = self.attn4(identity4, x)
        x = self.deconv2(x)
        # identity4 = F.resize(identity4, size=x.shape[2:])
        x = torch.cat((x, identity4), dim=1)
        x = self.double_conv6(x)
        
        if self.attn == 'yes':
          identity1 = self.attn1(identity1, x)             
        x = self.deconv3(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv7(x)
        
        x = self.final_conv(x)
        
        return self.out_activation(x)


class SubUnet(nn.Module):
    def __init__(self, n=0, in_channels=3, dimension=2, mhsa='none'):
        """
        This is the initial status of the sub-Unet.
        One can define his/her own subunet architectures.
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d

        self.n = n
        if self.n == 0:
          self.mhsa = mhsa
          self.double_conv2 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)
          self.double_conv3 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)

          self.double_conv4 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
          
          self.max_pool = max_pool
          
          self.deconv1 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)

          self.attn2 = Attn(16, dimension=dimension)

          self.mhsa1 = MHSA(32, dimension=dimension)
          self.final_conv = conv(in_channels=16, out_channels=1, kernel_size=1)
          self.out_activation = nn.Sigmoid()

        else:
          self.mhsa = mhsa
          self.double_conv2 = DoubleConvLayer(in_channels=16*(2**(n-1)), out_channels=32*(2**(n-1)), dimension=dimension)
          self.double_conv3 = DoubleConvLayer(in_channels=32*(2**(n-1)), out_channels=64*(2**(n-1)), dimension=dimension)

          self.double_conv4 = DoubleConvLayer(in_channels=64*(2**(n-1)), out_channels=32*(2**(n-1)), dimension=dimension)
          
          self.max_pool = max_pool
          
          self.deconv1 = convt(in_channels=64*(2**(n-1)), out_channels=32*(2**(n-1)), kernel_size=2, stride=2)

          self.attn2 = Attn(32*(2**(n-1)), dimension=dimension)

          self.mhsa1 = MHSA(64*(2**(n-1)), dimension=dimension)
        
    def forward(self, x):
        if self.n == 0:
          x = self.double_conv2(x)
          identity2 = x
          x = self.max_pool(x)

          x = self.double_conv3(x)
          if self.mhsa == 'yes':
            x = self.mhsa1(x)
          
          identity2 = self.attn2(identity2, x)
          x = self.deconv1(x)
          # identity2 = F.resize(identity2, size=x.shape[2:])
          x = torch.cat((x, identity2), dim=1)
          x = self.double_conv4(x)
          x = self.final_conv(x)
          x = self.out_activation(x)
        else:
          x = self.double_conv2(x)
          identity2 = x
          x = self.max_pool(x)

          x = self.double_conv3(x)
          if self.mhsa == 'yes':
            x = self.mhsa1(x)
          
          identity2 = self.attn2(identity2, x)
          x = self.deconv1(x)
          # identity2 = F.resize(identity2, size=x.shape[2:])
          x = torch.cat((x, identity2), dim=1)
          x = self.double_conv4(x)
        
        return x


class JoinUnet(nn.Module):
    def __init__(self, subunet, n, in_channels=3, dimension=2):
        """
        This is a class for combining two subunets with a convolutional layer.
        The two subunets are created based on the depth n, and then joined with another sub-structure to make it iterable.
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d

        self.n = n

        self.subunet1 = subunet
        self.subunet2 = subunet
        if self.n == 0:
          self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)
          self.double_conv2 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
          self.deconv1 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)
          self.final_conv = conv(in_channels=16, out_channels=1, kernel_size=1)
          self.max_pool = max_pool
          self.out_activation = nn.Sigmoid()
          self.attn1 = Attn(16, dimension=dimension)
        else:
          self.double_conv1 = DoubleConvLayer(in_channels=16*(2**(n-1)), out_channels=32*(2**(n-1)), dimension=dimension)
          self.double_conv2 = DoubleConvLayer(in_channels=64*(2**(n-1)), out_channels=32*(2**(n-1)), dimension=dimension)
          self.deconv1 = convt(in_channels=64*(2**(n-1)), out_channels=32*(2**(n-1)), kernel_size=2, stride=2)
          self.connect = conv(in_channels=64*(2**(n-1)), out_channels=32*(2**(n-1)), kernel_size=1)
          self.max_pool = max_pool
          self.attn1 = Attn(32*(2**(n-1)), dimension=dimension)
        
        # self.final_conv = conv(in_channels=16, out_channels=1, kernel_size=1)
        # self.out_activation = nn.Sigmoid()

    def forward(self, x):
        if self.n == 0:
          x = self.double_conv1(x)
          identity1 = x
          x = self.max_pool(x)
          x = self.subunet1(x)

          identity1 = self.attn1(identity1, x)           
          x = self.deconv1(x)
          # identity1 = F.resize(identity1, size=x.shape[2:])
          x = torch.cat((x, identity1), dim=1)
          x = self.double_conv2(x)
          
          x = self.final_conv(x)
          x = self.out_activation(x)
        else:
          x = self.double_conv1(x)
          identity1 = x
          x = self.max_pool(x)
          x = self.subunet1(x)
          x = self.connect(x)
          x = self.subunet2(x)

          identity1 = self.attn1(identity1, x)           
          x = self.deconv1(x)
          # identity1 = F.resize(identity1, size=x.shape[2:])
          x = torch.cat((x, identity1), dim=1)
          x = self.double_conv2(x)

        return x


class Frac(nn.Module):
    """
    This class is to create a fractal network structure for a given depth n.
    """
    def __init__(self, n: int, in_channels=3, dimension=2):
        super().__init__()
        self.subunet = SubUnet(n, in_channels=in_channels, dimension=dimension)
        self.n = n
        self.n_iter = n
        if self.n == 0:
          self.f_next = self.subunet
        else:
          self.f_next = JoinUnet(self.subunet, self.n_iter-1, in_channels=in_channels, dimension=dimension)
        self.n_iter -= 1
        while self.n_iter >= 1:
          self.f_next = JoinUnet(self.f_next, self.n_iter-1, in_channels=in_channels, dimension=dimension)
          self.n_iter -= 1

    def forward(self, x):
        x = self.f_next(x)
        return x


class AFUNet_0(nn.Module):
    """
    An example of f_1.
    Claim the dimension 2 as 2d or 3 as 3d.
    mhsa is the parameter controlling whether to use self-attention at the bottlenecks or not.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2, attn='yes', mhsa='none', dropout='none'):
        """
        
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d
        
        self.in_channels = in_channels
        self.mhsa = mhsa
        self.attn = attn
        self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)
        self.double_conv2 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)
        self.double_conv3 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)

        self.max_pool = max_pool
        self.out_activation = nn.Sigmoid()
 
        self.deconv1 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.final_conv = conv(in_channels=16, out_channels=out_channels, kernel_size=1)

        if self.attn == 'yes':
          self.attn1 = Attn(16, dimension=dimension)

        if self.mhsa == 'yes':
          self.mhsa1 = MHSA(32, dimension=dimension)
        
    def forward(self, x):
        
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        if self.mhsa == 'yes':
          x = self.mhsa1(x)

        if self.attn == 'yes':
          identity1 = self.attn1(identity1, x)
        x = self.deconv1(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv3(x)
        
        x = self.final_conv(x)
        
        return self.out_activation(x)


class AFUNet_1(nn.Module):
    """
    An example of f_1.
    Claim the dimension 2 as 2d or 3 as 3d.
    mhsa is the parameter controlling whether to use self-attention at the bottlenecks or not.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2, attn='yes', mhsa='none', dropout='none'):
        """
        
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d
        
        self.in_channels = in_channels
        self.mhsa = mhsa
        self.attn = attn
        self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)
        self.double_conv2 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)
        self.double_conv3 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)

        self.double_conv4 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)
        self.double_conv5 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
        self.double_conv6 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)

        self.double_conv7 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)
        self.double_conv8 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)

        self.double_conv9 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)

        self.max_pool = max_pool
        self.out_activation = nn.Sigmoid()

        
        self.deconv1 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv2 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv3 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.final_conv = conv(in_channels=16, out_channels=out_channels, kernel_size=1)

        if self.attn == 'yes':
          self.attn1 = Attn(32, dimension=dimension)
          self.attn2 = Attn(32, dimension=dimension)
          self.attn3 = Attn(16, dimension=dimension)

        if self.mhsa == 'yes':
          self.mhsa1 = MHSA(64, dimension=dimension)
          self.mhsa2 = MHSA(64, dimension=dimension)
        
    def forward(self, x):
        
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)

        x = self.double_conv3(x)
        if self.mhsa == 'yes':
          x = self.mhsa1(x)

        if self.attn == 'yes':
          identity2 = self.attn1(identity2, x)
        x = self.deconv1(x)
        # identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv4(x)
        x = self.double_conv5(x)
        x = self.double_conv6(x)
        identity3 = x
        x = self.max_pool(x)

        x = self.double_conv7(x)
        if self.mhsa == 'yes':
          x = self.mhsa2(x)
        
        if self.attn == 'yes':
          identity3 = self.attn2(identity3, x)
        x = self.deconv2(x)
        # identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv8(x)
        
        if self.attn == 'yes':
          identity1 = self.attn3(identity1, x)
        x = self.deconv3(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv9(x)
        
        x = self.final_conv(x)
        
        return self.out_activation(x)


class AFUNet_2(nn.Module):
    """
    An example of f_1.
    Claim the dimension 2 as 2d or 3 as 3d.
    mhsa is the parameter controlling whether to use self-attention at the bottlenecks or not.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2, attn='yes', mhsa='none', dropout='none'):
        """
        
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d
        
        self.in_channels = in_channels
        self.mhsa = mhsa
        self.attn = attn
        self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)

        self.double_conv2 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)
        self.double_conv3 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)

        self.double_conv4 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)

        self.double_conv5 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)
        self.double_conv6 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)

        self.double_conv7 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)

        self.double_conv8 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv9 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)
        self.double_conv10 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv11 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)

        self.double_conv12 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)
        self.double_conv13 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)
        self.double_conv14 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv15 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)

        self.double_conv16 = DoubleConvLayer(in_channels=64, out_channels=32, dimension=dimension)
        self.double_conv17 = DoubleConvLayer(in_channels=32, out_channels=64, dimension=dimension)
        self.double_conv18 = DoubleConvLayer(in_channels=64, out_channels=128, dimension=dimension)
        self.double_conv19 = DoubleConvLayer(in_channels=128, out_channels=64, dimension=dimension)

        self.double_conv20 = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
        self.double_conv21 = DoubleConvLayer(in_channels=16, out_channels=32, dimension=dimension)

        self.max_pool = max_pool
        self.out_activation = nn.Sigmoid()

        
        self.deconv1 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv2 = convt(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv3 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.deconv4 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv5 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv6 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv7 = convt(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.final_conv = conv(in_channels=16, out_channels=out_channels, kernel_size=1)

        if self.attn == 'yes':
          self.attn4 = Attn(32, dimension=dimension)
          self.attn2 = Attn(32, dimension=dimension)
          self.attn1 = Attn(16, dimension=dimension)

          self.attn3 = Attn(64, dimension=dimension)
          self.attn5 = Attn(64, dimension=dimension)
          self.attn6 = Attn(64, dimension=dimension)
          self.attn7 = Attn(64, dimension=dimension)

        if self.mhsa == 'yes':
          self.mhsa1 = MHSA(128, dimension=dimension)
          self.mhsa2 = MHSA(128, dimension=dimension)
          self.mhsa3 = MHSA(128, dimension=dimension)
          self.mhsa4 = MHSA(128, dimension=dimension)
        
    def forward(self, x):
        
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)

        x = self.double_conv3(x)
        identity3 = x
        x = self.max_pool(x)

        x = self.double_conv8(x)
        if self.mhsa == 'yes':
          x = self.mhsa1(x)

        if self.attn == 'yes':
          identity3 = self.attn3(identity3, x)
        x = self.deconv4(x)
        # identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv9(x)

        x = self.double_conv12(x)

        x = self.double_conv13(x)
        identity6 = x
        x = self.max_pool(x)

        x = self.double_conv14(x)
        if self.mhsa == 'yes':
          x = self.mhsa3(x)

        if self.attn == 'yes':
          identity6 = self.attn6(identity6, x)
        x = self.deconv6(x)
        # identity6 = F.resize(identity6, size=x.shape[2:])
        x = torch.cat((x, identity6), dim=1)
        x = self.double_conv15(x)
        
        if self.attn == 'yes':
          identity2 = self.attn2(identity2, x)
        x = self.deconv1(x)
        # identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv4(x)
        
        # x = self.double_conv20(x)
        # x = self.double_conv21(x)

        identity4 = x
        x = self.max_pool(x)

        x = self.double_conv5(x)
        identity5 = x
        x = self.max_pool(x)

        x = self.double_conv10(x)
        if self.mhsa == 'yes':
          x = self.mhsa2(x)

        if self.attn == 'yes':
          identity5 = self.attn3(identity5, x)
        x = self.deconv5(x)
        # identity5 = F.resize(identity5, size=x.shape[2:])
        x = torch.cat((x, identity5), dim=1)
        x = self.double_conv11(x)

        x = self.double_conv16(x)

        x = self.double_conv17(x)
        identity7 = x
        x = self.max_pool(x)

        x = self.double_conv18(x)
        if self.mhsa == 'yes':
          x = self.mhsa4(x)

        if self.attn == 'yes':
          identity7 = self.attn7(identity7, x)
        x = self.deconv7(x)
        # identity7 = F.resize(identity7, size=x.shape[2:])
        x = torch.cat((x, identity7), dim=1)
        x = self.double_conv19(x)
        
        if self.attn == 'yes':
          identity4 = self.attn4(identity4, x)
        x = self.deconv2(x)
        # identity4 = F.resize(identity4, size=x.shape[2:])
        x = torch.cat((x, identity4), dim=1)
        x = self.double_conv6(x)
        
        if self.attn == 'yes':
          identity1 = self.attn1(identity1, x)             
        x = self.deconv3(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv7(x)
        
        x = self.final_conv(x)
        
        return self.out_activation(x)


class AFUNet_3(nn.Module):
    """
    An example of f_1.
    Claim the dimension 2 as 2d or 3 as 3d.
    mhsa is the parameter controlling whether to use self-attention at the bottlenecks or not.
    """
    def __init__(self, in_channels: int, out_channels: int, dimension=2, attn='yes', mhsa='none', dropout='none'):
        """
        
        """
        super().__init__()

        if dimension == 2:
          conv = nn.Conv2d
          max_pool = nn.MaxPool2d(kernel_size=(2, 2))
          convt = nn.ConvTranspose2d
        elif dimension == 3:
          conv = nn.Conv3d
          max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
          convt = nn.ConvTranspose3d
        
        self.in_channels = in_channels
        self.mhsa = mhsa
        self.attn = attn
        self.double_conv0 = DoubleConvLayer(in_channels=in_channels, out_channels=16, dimension=dimension)
        
        #

        self.double_conv1 = DoubleConvLayer(in_channels=16, out_channels=16*2, dimension=dimension)

        self.double_conv2 = DoubleConvLayer(in_channels=16*2, out_channels=32*2, dimension=dimension)
        self.double_conv3 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)

        self.double_conv4 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)

        self.double_conv5 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv6 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)

        self.double_conv7 = DoubleConvLayer(in_channels=32*2, out_channels=16*2, dimension=dimension)

        self.double_conv8 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv9 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)
        self.double_conv10 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv11 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)

        self.double_conv12 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)
        self.double_conv13 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv14 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv15 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)

        self.double_conv16 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)
        self.double_conv17 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv18 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv19 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)
           
        self.deconv1 = convt(in_channels=64*2, out_channels=32*2, kernel_size=2, stride=2)
        self.deconv2 = convt(in_channels=64*2, out_channels=32*2, kernel_size=2, stride=2)
        self.deconv3 = convt(in_channels=32*2, out_channels=16*2, kernel_size=2, stride=2)

        self.deconv4 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv5 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv6 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv7 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)

        #
        
        self.double_conv1_1 = DoubleConvLayer(in_channels=16, out_channels=16*2, dimension=dimension)

        self.double_conv2_1 = DoubleConvLayer(in_channels=16*2, out_channels=32*2, dimension=dimension)
        self.double_conv3_1 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)

        self.double_conv4_1 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)

        self.double_conv5_1 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv6_1 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)

        self.double_conv7_1 = DoubleConvLayer(in_channels=32*2, out_channels=16*2, dimension=dimension)

        self.double_conv8_1 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv9_1 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)
        self.double_conv10_1 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv11_1 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)

        self.double_conv12_1 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)
        self.double_conv13_1 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv14_1 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv15_1 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)

        self.double_conv16_1 = DoubleConvLayer(in_channels=64*2, out_channels=32*2, dimension=dimension)
        self.double_conv17_1 = DoubleConvLayer(in_channels=32*2, out_channels=64*2, dimension=dimension)
        self.double_conv18_1 = DoubleConvLayer(in_channels=64*2, out_channels=128*2, dimension=dimension)
        self.double_conv19_1 = DoubleConvLayer(in_channels=128*2, out_channels=64*2, dimension=dimension)
           
        self.deconv1_1 = convt(in_channels=64*2, out_channels=32*2, kernel_size=2, stride=2)
        self.deconv2_1 = convt(in_channels=64*2, out_channels=32*2, kernel_size=2, stride=2)
        self.deconv3_1 = convt(in_channels=32*2, out_channels=16*2, kernel_size=2, stride=2)

        self.deconv4_1 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv5_1 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv6_1 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        self.deconv7_1 = convt(in_channels=128*2, out_channels=64*2, kernel_size=2, stride=2)
        #
        self.deconv0 = convt(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.double_conv_final = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
        self.connection = DoubleConvLayer(in_channels=32, out_channels=16, dimension=dimension)
        self.final_conv = conv(in_channels=16, out_channels=out_channels, kernel_size=1)
        self.max_pool = max_pool
        self.out_activation = nn.Sigmoid()

        if self.attn == 'yes':
          self.attn4 = Attn(32*2, dimension=dimension)
          self.attn2 = Attn(32*2, dimension=dimension)
          self.attn1 = Attn(16*2, dimension=dimension)
          self.attn3 = Attn(64*2, dimension=dimension)
          self.attn5 = Attn(64*2, dimension=dimension)
          self.attn6 = Attn(64*2, dimension=dimension)
          self.attn7 = Attn(64*2, dimension=dimension)
          #
          self.attn4_1 = Attn(32*2, dimension=dimension)
          self.attn2_1 = Attn(32*2, dimension=dimension)
          self.attn1_1 = Attn(16*2, dimension=dimension)
          self.attn3_1 = Attn(64*2, dimension=dimension)
          self.attn5_1 = Attn(64*2, dimension=dimension)
          self.attn6_1 = Attn(64*2, dimension=dimension)
          self.attn7_1 = Attn(64*2, dimension=dimension)

          self.attn0 = Attn(16, dimension=dimension)

        if self.mhsa == 'yes':
          self.mhsa1 = MHSA(128*2, dimension=dimension)
          self.mhsa2 = MHSA(128*2, dimension=dimension)
          self.mhsa3 = MHSA(128*2, dimension=dimension)
          self.mhsa4 = MHSA(128*2, dimension=dimension)
          #
          self.mhsa1_1 = MHSA(128*2, dimension=dimension)
          self.mhsa2_1 = MHSA(128*2, dimension=dimension)
          self.mhsa3_1 = MHSA(128*2, dimension=dimension)
          self.mhsa4_1 = MHSA(128*2, dimension=dimension)
        
    def forward(self, x):

        x = self.double_conv0(x)
        identity0 = x
        x = self.max_pool(x)

        #
        
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)

        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)

        x = self.double_conv3(x)
        identity3 = x
        x = self.max_pool(x)

        x = self.double_conv8(x)
        if self.mhsa == 'yes':
          x = self.mhsa1(x)

        if self.attn == 'yes':
          identity3 = self.attn3(identity3, x)
        x = self.deconv4(x)
        # identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv9(x)

        x = self.double_conv12(x)

        x = self.double_conv13(x)
        identity6 = x
        x = self.max_pool(x)

        x = self.double_conv14(x)
        if self.mhsa == 'yes':
          x = self.mhsa3(x)

        if self.attn == 'yes':
          identity6 = self.attn6(identity6, x)
        x = self.deconv6(x)
        # identity6 = F.resize(identity6, size=x.shape[2:])
        x = torch.cat((x, identity6), dim=1)
        x = self.double_conv15(x)
        
        if self.attn == 'yes':
          identity2 = self.attn2(identity2, x)
        x = self.deconv1(x)
        # identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv4(x)
        identity4 = x
        x = self.max_pool(x)

        x = self.double_conv5(x)
        identity5 = x
        x = self.max_pool(x)

        x = self.double_conv10(x)
        if self.mhsa == 'yes':
          x = self.mhsa2(x)

        if self.attn == 'yes':
          identity5 = self.attn3(identity5, x)
        x = self.deconv5(x)
        # identity5 = F.resize(identity5, size=x.shape[2:])
        x = torch.cat((x, identity5), dim=1)
        x = self.double_conv11(x)

        x = self.double_conv16(x)

        x = self.double_conv17(x)
        identity7 = x
        x = self.max_pool(x)

        x = self.double_conv18(x)
        if self.mhsa == 'yes':
          x = self.mhsa4(x)

        if self.attn == 'yes':
          identity7 = self.attn7(identity7, x)
        x = self.deconv7(x)
        # identity7 = F.resize(identity7, size=x.shape[2:])
        x = torch.cat((x, identity7), dim=1)
        x = self.double_conv19(x)
        
        if self.attn == 'yes':
          identity4 = self.attn4(identity4, x)
        x = self.deconv2(x)
        # identity4 = F.resize(identity4, size=x.shape[2:])
        x = torch.cat((x, identity4), dim=1)
        x = self.double_conv6(x)

        if self.attn == 'yes':
          identity1 = self.attn1(identity1, x)             
        x = self.deconv3(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv7(x)

        # 
        x = self.connection(x)
        #
        
        x = self.double_conv1_1(x)
        identity1_1 = x
        x = self.max_pool(x)

        x = self.double_conv2_1(x)
        identity2_1 = x
        x = self.max_pool(x)

        x = self.double_conv3_1(x)
        identity3_1 = x
        x = self.max_pool(x)

        x = self.double_conv8_1(x)
        if self.mhsa == 'yes':
          x = self.mhsa1_1(x)

        if self.attn == 'yes':
          identity3_1 = self.attn3_1(identity3_1, x)
        x = self.deconv4_1(x)
        # identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3_1), dim=1)
        x = self.double_conv9_1(x)

        x = self.double_conv12_1(x)

        x = self.double_conv13_1(x)
        identity6_1 = x
        x = self.max_pool(x)

        x = self.double_conv14_1(x)
        if self.mhsa == 'yes':
          x = self.mhsa3_1(x)

        if self.attn == 'yes':
          identity6_1 = self.attn6(identity6_1, x)
        x = self.deconv6_1(x)
        # identity6 = F.resize(identity6, size=x.shape[2:])
        x = torch.cat((x, identity6_1), dim=1)
        x = self.double_conv15_1(x)
        
        if self.attn == 'yes':
          identity2_1 = self.attn2_1(identity2_1, x)
        x = self.deconv1_1(x)
        # identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2_1), dim=1)
        x = self.double_conv4_1(x)
        identity4_1 = x
        x = self.max_pool(x)

        x = self.double_conv5_1(x)
        identity5_1 = x
        x = self.max_pool(x)

        x = self.double_conv10_1(x)
        if self.mhsa == 'yes':
          x = self.mhsa2_1(x)

        if self.attn == 'yes':
          identity5_1 = self.attn3_1(identity5_1, x)
        x = self.deconv5_1(x)
        # identity5 = F.resize(identity5, size=x.shape[2:])
        x = torch.cat((x, identity5_1), dim=1)
        x = self.double_conv11_1(x)

        x = self.double_conv16_1(x)

        x = self.double_conv17_1(x)
        identity7_1 = x
        x = self.max_pool(x)

        x = self.double_conv18_1(x)
        if self.mhsa == 'yes':
          x = self.mhsa4_1(x)

        if self.attn == 'yes':
          identity7_1 = self.attn7_1(identity7_1, x)
        x = self.deconv7_1(x)
        # identity7 = F.resize(identity7, size=x.shape[2:])
        x = torch.cat((x, identity7_1), dim=1)
        x = self.double_conv19_1(x)
        
        if self.attn == 'yes':
          identity4_1 = self.attn4_1(identity4_1, x)
        x = self.deconv2_1(x)
        # identity4 = F.resize(identity4, size=x.shape[2:])
        x = torch.cat((x, identity4_1), dim=1)
        x = self.double_conv6_1(x)

        if self.attn == 'yes':
          identity1_1 = self.attn1_1(identity1_1, x)             
        x = self.deconv3_1(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1_1), dim=1)
        x = self.double_conv7_1(x)

        #

        if self.attn == 'yes':
          identity0 = self.attn0(identity0, x)             
        x = self.deconv0(x)
        # identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity0), dim=1)
        x = self.double_conv_final(x)

        x = self.final_conv(x)
        
        return self.out_activation(x)


        
