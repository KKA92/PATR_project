import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from bintorch import *
from torchsummary import summary
# Device configuration
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = (3,32,32) # 28x28

conv_filter1 = 128
conv_filter2 = 256
conv_filter3 = 512

linear_size = 1024

num_class = 10

num_epochs = 10 
batch_size = 50
learning_rate = 0.003

bin_mode = False
batch_enable = True

# CIFAR-10 dataset 
train_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

#examples = iter(test_loader)
#example_data, example_targets = examples.next()

#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

def _init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        init.kaiming_normal_(m.weight)



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        #self.input_size = input_size
        self.conv128_1 = B_Conv2d(3, conv_filter1, 3, stride=1, padding=1, binary=bin_mode) # (32-3+(2*1)) / 1 + 1 = 32 
        self.relu128_1 = nn.ReLU()
        self.bn128_1 = nn.BatchNorm2d(conv_filter1)
        self.conv128_2 = B_Conv2d(conv_filter1,conv_filter1,3 ,stride=1, padding=1, binary=bin_mode)
        self.relu128_2 = nn.ReLU()
        self.bn128_2 = nn.BatchNorm2d(conv_filter1)
        self.mp128 = nn.MaxPool2d(2,2) #window = 2 , step = 2  / 16 x 16 
        self.bn128_3 = nn.BatchNorm2d(conv_filter1)
		
        self.conv256_1 = B_Conv2d(conv_filter1, conv_filter2, 3, stride=1, padding=1, binary=bin_mode) # (16-3+(2*1)) / 1 + 1 = 16 
        self.relu256_1 = nn.ReLU()
        self.bn256_1 = nn.BatchNorm2d(conv_filter2)
        self.conv256_2 = B_Conv2d(conv_filter2, conv_filter2,3 ,stride=1, padding=1, binary=bin_mode)
        self.relu256_2 = nn.ReLU()
        self.bn256_2 = nn.BatchNorm2d(conv_filter2)
        self.mp256 = nn.MaxPool2d(2,2) #window = 2 , step = 2  / 8 x 8 
        self.bn256_3 = nn.BatchNorm2d(conv_filter2)
        
        self.conv512_1 = B_Conv2d(conv_filter2,conv_filter3, 3, stride=1, padding=1, binary=bin_mode) # (8-3+(2*1)) / 1 + 1 = 8
        self.relu512_1 = nn.ReLU()
        self.bn512_1 = nn.BatchNorm2d(conv_filter3)
        self.conv512_2 = B_Conv2d(conv_filter3, conv_filter3,3 ,stride=1, padding=1, binary=bin_mode)
        self.relu512_2 = nn.ReLU()
        self.bn512_2 = nn.BatchNorm2d(conv_filter3)
        self.mp512 = nn.MaxPool2d(2,2) #window = 2 , step = 2  / 4 x 4 
        self.bn512_3 = nn.BatchNorm2d(conv_filter3)

        self.linear1024_1 = B_Linear(512*4*4,linear_size,binary=bin_mode)
        self.relu1024_1 = nn.ReLU()
        self.bn1024_1 = nn.BatchNorm1d(linear_size)
        self.linear1024_2 = B_Linear(linear_size,linear_size,binary=bin_mode)
        self.relu1024_2 = nn.ReLU()
        self.bn1024_2 = nn.BatchNorm1d(linear_size)
        
        self.linear10_1 = B_Linear(linear_size, num_class, binary=bin_mode)
        self.relu10_1 = nn.ReLU()
        self.bn10_1 = nn.BatchNorm1d(num_class)

        self.apply(_init_weights) # init_ 'w' of kaiming he method
    def forward(self, x):
        if batch_enable == True :
            out = self.bn128_1(self.relu128_1(self.conv128_1(x)))
            out = self.bn128_3(self.mp128(self.bn128_2(self.relu128_2(self.conv128_2(out)))))

            out = self.bn256_1(self.relu256_1(self.conv256_1(out)))
            out = self.bn256_3(self.mp256(self.bn256_2(self.relu256_2(self.conv256_2(out)))))
        
            out = self.bn512_1(self.relu512_1(self.conv512_1(out)))
            out = self.bn512_3(self.mp512(self.bn512_2(self.relu512_2(self.conv512_2(out)))))
            out = out.view(-1, 512*4*4)
            out = self.bn1024_1(self.relu1024_1(self.linear1024_1(out)))
            out = self.bn1024_2(self.relu1024_2(self.linear1024_2(out)))

            out = self.bn10_1(self.relu10_1(self.linear10_1(out)))
            return out
        else : 
            out = self.relu128_1(self.conv128_1(x))
            out = self.mp128(self.relu128_2(self.conv128_2(out)))

            out = self.relu256_1(self.conv256_1(out))
            out = self.mp256(self.relu256_2(self.conv256_2(out)))
        
            out = self.relu512_1(self.conv512_1(out))
            out = self.mp512(self.relu512_2(self.conv512_2(out)))
            out = out.view(-1, 512*4*4)
            out = self.relu1024_1(self.linear1024_1(out))
            out = self.relu1024_2(self.linear1024_2(out))

            out = self.relu10_1(self.linear10_1(out))
            return out
            

model = NeuralNet(input_size, num_class).to(device)
#print (model)
#name_list = []
name_list = [x for x in model.named_parameters() if "conv" in x or "linear" in x]
'''
for name, param in model.named_parameters():
    if 'conv' in name : 
        if 'weight' in name : 
            name_list.append(name)
#            print(name,param.weight.data)
    elif 'linear' in name :
        if 'weight' in name :
            name_list.append(name)
#            print(name,param.weight.data)
'''	
	#print ('name = \n',name,'param = \n', param.data.shape)
#print (name_list)
#summary(model,(1,28,28) ,batch_size=batch_size)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
#steps = (0.000002/ 0.003)**(1./num_epochs)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
layer_shape = []
for name , param in model.named_parameters():
    layer_shape.append(param.data.shape)
#print (layer_shape)
# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        #images = images.reshape(-1, 28*28).to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # grad flush 
        loss.backward() # back propagation 
        
        for name, param in model.named_parameters():
            #clipping 'gt'
            #print(param.data.shape)
            #if name in ['l1.weight', 'l2.weight', 'l3.weight', 'l4.weight']:
            if name in name_list:
                if bin_mode == True:
                    param.data = torch.sign(param.data)
                #print (name, param)
        optimizer.step() # update param 
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, num_epochs,i+1,n_total_steps, loss.item()))
    scheduler.step()
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    #print(f'Accuracy of the network on the 10000 test images: {acc} %')
    print('Accuracy of the network on the 10000 test images: %s'%(acc))

#for name, param in model.named_parameters():
#	print ('name = \n',name,'param = \n', param.data)

#print (model)
#activation = []
#for name, param in model.named_parameters():
#    if name in ['l1.weight', 'l2.weight', 'l3.weight', 'l4.weight']:
#        activation.append(param.data)

activation = {}
def get_activation(name):
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook

model.conv128_1.register_forward_hook(get_activation('conv128_1'))
model.conv256_1.register_forward_hook(get_activation('conv256_1'))
model.conv512_1.register_forward_hook(get_activation('conv512_1'))
examples = iter(test_loader)
example_data, example_targets = examples.next()

#print ('data.shape',example_data[0].shape)
data = example_data.squeeze().to(device) # squeez of 'number of batch idx' & send to 'cuda' for process image

output = model(data)

act1 = activation['conv128_1'].squeeze().cpu()
act2 = activation['conv256_1'].squeeze().cpu()
act3 = activation['conv512_1'].squeeze().cpu()
#print ('act',act.shape)

#print ('act[0]',act[0].shape)

for idx in range(12) : #layer1 filter channel 
    plt.subplot(3, 4, idx+1)
    if (idx < 4): # 0 ~ 3 is  conv128 layer 
        plt.imshow(act1[0][idx]) # 32 x 32
    elif (idx < 8) and (idx >= 4) : # 4 ~ 7 is conv256 layer
	    plt.imshow(act2[0][idx]) # 16 x 16
    else : # 8~ 11 is conv 512 layer
	    plt.imshow(act3[0][idx]) # 8 x 8
#fig, axarr = plt.subplots(act.size(0))
#print (idx,act.shape)
#for idx in range(act.size(0)):
#    axarr[idx].imshow(act[idx])

#print (model.conv128_1.weight.data.shape)

#show_features = torch.tensor(model.l1.weight.data).reshape(784,28,28)
#print (show_features.shape)
#for i in range(10):
    #print(model.l4.weight.data[i])
#    plt.subplot(5,5,i+1)
#    plt.imshow(model.conv128_1.weight.data[i].to('cpu'))
    #plt.imshow(model.l4.weight.data[i].to('cpu'))
plt.show()

