import torch

device = "cuda"

x = torch.tensor([
    [6.0], [8.0], [10.0],
    [12.0], [14.0]])

y = torch.tensor([
    [10.0], [12.0], [14.0],
    [16.0], [18.0]])
x = x.to(device)
y = y.to(device)
#The relation between data is just x*1 + 4

model = torch.nn.Linear(1,1).to(device) #one feature, one output

lossRSE = torch.nn.MSELoss() 

optimizer =  torch.optim.SGD(model.parameters(), lr=0.0001) #learning rate = 0.001

def singleIteration() -> float:
    prediction = model(x)
    loss = lossRSE(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item() #this is slow because it goes back to the cpu

def trainingRun(amoutOfRuns : int):
    for i in range(amoutOfRuns):
        loss = singleIteration()
        if i % 500 == 0: #one epoche is 500
            print(f'Iteration{i}: Loss={loss}')
    print("Weight:", model.weight.item())
    print("Bias:", model.bias.item())

trainingRun(1000000)
#After letting it run once with 1000000 iterations on cpu:
#Weight: 1.0007818937301636
#Bias: 3.9915852546691895
#For this work Cpu is way faster because we are using such a small model and Gpu's are faster when calculating much more