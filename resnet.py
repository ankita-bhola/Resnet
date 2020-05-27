import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional
import math

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = models.resnet18(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()
PATH='./resnet.pth'

for param in model.parameters():
    param.requires_grad=False
model.fc=nn.Linear(512,10)
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

bound=1/math.sqrt(model.fc.weight.size(1))
model.fc.weight.data.uniform_(-bound,bound)
model.fc.bias.data.uniform_(-bound,bound)

for epoch in range(2):
    running_loss=0
    batch_loss=0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        if(i%500==0):
            print(batch_loss)
            batch_loss=0
        optimizer.step()
        running_loss+=loss.item()
        batch_loss+=loss.item()
      
        
torch.save(model.state_dict(),PATH)

model.load_state_dict(torch.load(PATH))
total=0
correct=0
with torch.no_grad():
    for data in testloader:
        images,labels= data
        outputs = model(images)
        _,predicted=torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
