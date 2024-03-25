import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *
import pdb
#################################
# U-net for speech dereverberation
# base implementation on http://github.com//milesial/Pytorch-UNet
#################################

class LateSupUnet(nn.Module):
	def __init__(self, n_channels=1, bilinear=True):
		super(LateSupUnet, self).__init__()
		self.n_channels = n_channels
		self.bilinear = bilinear
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, 1)
                
	def forward(self, input):
        #pdb.set_trace()
        #input = input.unsqueeze(0)
		x1 = self.inc(input)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2) 
		x = self.up4(x, x1)
		x = self.outc(x)
		output = (input - x)     #.squeeze(0).squeeze(0)
		return output

class Channel_select(nn.Module):
    def __init__(self):
        super(Channel_select, self).__init__()
        self.l1 = nn.Linear( 65, 1024 )
        self.l2 = nn.Linear( 1024, 512 )
        self.l3 = nn.Linear( 512, 256 )
        self.l4 = nn.Linear( 256, 22 )
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = input.permute(0, 2, 1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = x.permute(0, 2, 1)
        x = self.NofM_custom(x, 8)
        
        return x

    def NofM_custom(self, x, n):
        y = torch.zeros(x.shape).cuda()       
        val, ind = torch.topk(x, n, dim=1)
        #pdb.set_trace()
        y.scatter_(1, ind, val)   

        return y
	
class Deep_ElectroNet(nn.Module):
	def __init__(self):
		super(Deep_ElectroNet, self).__init__()
		self.LS_UNet = LateSupUnet(1, True)
		self.CS = Channel_select()

	def forward(self, input):
		x = self.LS_UNet(input)
		x = x.squeeze(0)
		#print("x ", x.shape)
		x = self.CS(x)
		x = x.squeeze(0)

		return x

