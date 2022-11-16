import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


#load
data = pd.read_csv('train3.csv')

# Data sets
X = data.iloc[:,:4002]
y = data.iloc[:,-1]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=101)

# Plot the data
#import matplotlib.pyplot as plt
#print(plt.plot(IRIS_TRAINING ,IRIS_TEST))


#Bulied the model
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

#hyperparameters
hl = 3
lr = 0.01
num_epoch = 1800

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4002, hl)
        self.fc2 = nn.Linear(hl, 2)
        #self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

for epoch in range(num_epoch):
    X = torch.Tensor(xtrain.values).float()
    Y = torch.Tensor(ytrain.values).long()

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    acc = 100 * torch.sum(Y==torch.max(out.data, 1)[1]).double() / len(Y)
    if (epoch % 50 == 1):
	    print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                   %(epoch+1, num_epoch, loss.item(), acc.item()))


#get prediction
X = torch.Tensor(xtest.values).float()
Y = torch.Tensor(ytest.values).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of testing %.4f %%' % (100 * torch.sum(Y==predicted).double() / len(Y)))
