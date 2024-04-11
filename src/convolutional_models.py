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
		#output = (input - x)     #.squeeze(0).squeeze(0)
		return x

class Channel_select(nn.Module):
    def __init__(self):
        super(Channel_select, self).__init__()
        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear( 64, 22)
        #self.l3 = nn.Linear( 512, 256 )
        #self.l4 = nn.Linear( 256, 22 )
        self.relu = nn.ReLU()
    
    def forward(self, input):
        #x = input.permute(0, 2, 1)
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
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
		self.down_sp = nn.ModuleList()
		self.down = nn.Conv1d(in_channels=3556, out_channels=2005, kernel_size=1, stride=1) #nn.Linear(3556, 2005)
		self.up = nn.Conv1d(in_channels=125, out_channels=3556, kernel_size=1, stride=1)
		
		for _ in range(2):
			self.down_sp.append(nn.Sequential(
				nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
				nn.ReLU(),
				nn.MaxPool1d(2)		
			))
		
		self.LS_UNet = LateSupUnet(1, True)
		self.CS = Channel_select()

	def forward(self, input):
		input = input.permute(0, 2, 1)
		input = self.down(input) #(1, 128, 3556) -> (1, 128, 2005)
		input = input.permute(0, 2, 1)
		for down_sampling in self.down_sp:
			input = down_sampling(input) #(1, 128, 3556) -> (1, 128, 125)

		input = input.unsqueeze(0) #(1, 128, 125) -> (1, 1, 128, 125)
		x1 = self.LS_UNet(input)
		x1 = x1.squeeze(0) #(1, 128, 125)
		x1 = x1.permute(0, 2, 1) #(1, 125, 128)
		x = self.up(x1) #(1, 3556, 128)
		print("x ", x.shape)
		output = self.CS(x)
		output = output.squeeze(0)

		return output, x1.squeeze(0)

