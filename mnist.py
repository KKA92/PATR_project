import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse


from bintorch import *
from torchsummary import summary
# argparse


parser = argparse.ArgumentParser(description='PyTorch MNIST Binaryconnect Training')

parser.add_argument('--optim', default='SGD', type=str, help='optimizer select')

parser.add_argument('--loss', default='MSE', type=str, help='loss function select')

parser.add_argument('--epochs', type=int, default=10,
                    help = 'number of epoch')

parser.add_argument('--train_scheme', type=str, default=False,
                    help = 'training scheme')

parser.add_argument('--print_freq', default=100, type=int , help='log print frequency')

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = args.epochs
batch_size = 100
learning_rate = 0.001

if args.train_scheme == 'True':
    bin_mode = True
else :
    bin_mode = False


print ('binary mode is ? ',bin_mode, args.train_scheme)

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data_mnist', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data_mnist', 
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

PATH = './mnist_checkpoint/'



# Loss and optimizer
if (args.loss == 'MSE') : 
    criterion = nn.MSELoss()
    print ('loss function is MSE')
else : 
    criterion = nn.CrossEntropyLoss()
    print ('loss function is CrossEntropy')

if (args.optim == 'SGD') : 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    print ('optim function is SGD')
else :
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    print ('optim function is Adam')

'''
model = torch.load(PATH+ 'model_mnist.pt')
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))
checkpoint = torch.load(PATH + 'all.tar')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
'''

layer_shape = []
for name , param in model.named_parameters():
    layer_shape.append(param.data.shape)
#print (layer_shape)
# Train the model
n_total_steps = len(train_loader)
loss_list = []
cycle_list = []
def train(train_loader,num_epochs): 

    for epoch in range(num_epochs):
        #for epoch in epoch_list:
        for i, (images, labels) in enumerate(train_loader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(images) 
            #print (outputs.shape)
            #print (labels.shape)
            if args.loss == 'MSE' : 
                one_hot = torch.nn.functional.one_hot(labels, 10).float()#mse
                loss = criterion(outputs, one_hot) # mse 
            else : 
                loss = criterion(outputs, labels) # cross entropy
        
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
            if (i+1) % args.print_freq == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, num_epochs,i+1,n_total_steps, loss.item()))
                if (i+1 == n_total_steps) : 
                    loss_list.append(loss.item())
				

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
acc_list = []
best_acc = 0.0
def test(test_loader): 
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
        #print('Accuracy of the network on the 10000 test images: %s'%(acc))
        #print(f'Accuracy of the network on the 10000 test images: {acc} %')
    return acc


for i in range(args.epochs) : 
    train(train_loader, 1)
    acc_list.append(test(test_loader))
    print('Accuracy of the network in %s  %s %s '%(i,acc_list[i],best_acc))
    if best_acc < acc_list[i] :
        best_acc = acc_list[i]
        torch.save(model, PATH + 'model_mnist.pt')
        torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
        torch.save({'model' : model.state_dict(),
        'optimizer': optimizer.state_dict()},PATH+'all.tar')
    cycle_list.append(i+1)

fig = plt.figure() # plt figure create 
fig.suptitle('MNIST'+args.optim+'_'+args.loss+'_'+str(args.train_scheme)+'_'+str(args.epochs)+'_graph.svg')
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(cycle_list, acc_list, color = 'blue', linewidth = 2)
ax1.set_ylabel('Acc'+str(best_acc))
ax2.plot(cycle_list, loss_list,color = 'red', linewidth = 2)
ax2.set_ylabel('loss')
ax2.set_xlabel('Epochs')
plt.savefig('mnist_'+args.optim+'_'+args.loss+'_'+str(args.train_scheme)+'_'+str(args.epochs)+'_graph.svg')

plt.clf()
#plt.show()
#print (model)
#activation = []
#for name, param in model.named_parameters():
#    if name in ['l1.weight', 'l2.weight', 'l3.weight', 'l4.weight']:
#        activation.append(param.data)


# feature map
#print (model.l1.weight.data.shape)
show_features =ccorch.tensor(model.l1.weight.data).reshape(784,28,28)
#print (show_features.shape)
for i in range(10):
    #print(model.l4.weight.data[i])
    plt.subplot(5,5,i+1)
    plt.imshow(show_features[i+40].to('cpu'), cmap ='gray')
    #plt.imshow(model.l4.weight.data[i].to('cpu'))
fig_name = 'mnist_'+args.optim+'_'+args.loss+'_'+str(args.train_scheme)+'_'+str(args.epochs)+'_featuremap.svg'
plt.suptitle(fig_name)
plt.savefig(fig_name,dpi=300)
#plt.show()

print ('Summury best acc %s'%(best_acc))

