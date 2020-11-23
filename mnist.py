import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from bintorch import *
from torchsummary import summary
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001


bin_mode = False
# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

def _init_weights(module):
    if type(module) == nn.Linear:
        init.kaiming_normal_(m.weight)



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #self.input_size = input_size
        self.l1 = B_Linear(input_size, input_size,binary=bin_mode)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.l2 = B_Linear(input_size, input_size,binary=bin_mode)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(input_size)
        self.l3 = B_Linear(input_size,input_size,binary=bin_mode)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(input_size)
        self.l4 = B_Linear(input_size,num_classes,binary=bin_mode)
        self.apply(_init_weights) # init_ 'w' of kaiming he method
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.l4(out)
        # no activation and no softmax at the end just Linear! 
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
#summary(model,(1,28,28) ,batch_size=batch_size)

#activation = {}
#def get_activation(name):
#    def hook(model, input, output):
#        activation[name] = output.detach()
#        return hook

#weights_l1 = model[0].weight.data.numpy()
#model[0].register_forward_hook(get_activation('l1'))


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

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
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # grad flush 
        loss.backward() # back propagation 
        optimizer.step() # update param 
        for name, param in model.named_parameters():
            #clipping 'gt'
            #print(param.data.shape)
            if name in ['l1.weight', 'l2.weight', 'l3.weight', 'l4.weight']:
                if bin_mode == True:
                    param.data = torch.sign(param.data)
                #print (name, param)
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, num_epochs,i+1,n_total_steps, loss.item()))
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
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
activation = []
#for name, param in model.named_parameters():
#    if name in ['l1.weight', 'l2.weight', 'l3.weight', 'l4.weight']:
#        activation.append(param.data)
print (model.l1.weight.data.shape)
show_features = torch.tensor(model.l1.weight.data).reshape(784,28,28)
print (show_features.shape)
for i in range(10):
    #print(model.l4.weight.data[i])
    plt.subplot(5,5,i+1)
    plt.imshow(show_features[i+40].to('cpu'), cmap ='gray')
    #plt.imshow(model.l4.weight.data[i].to('cpu'))
plt.show()
