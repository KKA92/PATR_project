import torch
import torchvision

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


def hard_sigmoid(x):
    return torch.clamp(0,1,(x+1)/2)

def binarization(W,binary=True, det=False, sto=False):

	#print("bin state det %d"%(det))
	
	Wb = W
	if binary == False or (det and sto):
		pass
	else : 
        
		if sto :
			r_v = torch.rand(1)
			Wb = hard_sigmoid(W)
			Wb[Wb >= r_v] = 1 # change if element Wb >= 0 : 1 
			Wb[Wb < r_v] = -1 # change if element Wb < 0 : -1
		else: #det
		#	print (Wb)
			Wb[Wb >= 0] = 1 # change if element Wb >= 0 : 1 
			Wb[Wb < 0] = -1 # change if element Wb < 0 : -1
			#print ('Quantized W ',Wb)
	return Wb # return tensor 


class B_Conv2d(nn.Conv2d):
	def __init__(self,in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, binary=True):
		super(B_Conv2d,self).__init__(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, stride= stride, padding=padding, dilation= dilation, groups= groups, bias=bias)
		self.binary = binary
	def forward(self, x, binary=False):
  
		if self.binary == True:
			bin_weight = binarization(self.weight.data, binary = self.binary, det=True, sto=False)
			#print('conv bin_weight', bin_weight)
			return F.conv2d(x, bin_weight, self.bias, self.padding, self.dilation, self.groups)
		else : 
			return F.conv2d(x, self.weight, self.bias, self.padding, self.dilation, self.groups)

class B_Linear(nn.Linear):
	def __init__(self,in_features, out_features, bias=True, binary=True):
		super(B_Linear,self).__init__(in_features=in_features, out_features=out_features, bias=bias)
		self.binary = binary

	def forward(self, x):
		if self.binary == True:
			weight_bin = binarization(self.weight.data , binary = self.binary, det=True, sto=False)
			#print('binary == True')
			#print('self.weight_linear',weight_bin)
			return F.linear(x, weight_bin, self.bias)
		else :
			#print ('binary == false!')
			return F.linear(x, self.weight, self.bias)
            

