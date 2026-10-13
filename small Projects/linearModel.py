import torch

x = torch.tensor([
    [6.0], [8.0], [10.0],
    [12.0], [14.0]])

y = torch.tensor([
    [10.0], [12.0], [14.0],
    [16.0], [18.0]])
#The relation between data is just x*1 + 4

model = torch.nn.Linear(1,1) #one feature, one output

lossRSE = torch.nn.MSELoss() 

optimizer =  torch.optim.SGD(model.parameters(), lr=0.0001) #learning rate = 0.001

def singleIteration() -> float:
    prediction = model(x)
    loss = lossRSE(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def trainingRun(amoutOfRuns : int):
    for i in range(amoutOfRuns):
        loss = singleIteration()
        if i % 500 == 0: #one epoche is 500
            print(f'Iteration{i}: Loss={loss}')
    print("Weight:", model.weight.item())
    print("Bias:", model.bias.item())

trainingRun(1000000)
#Weight: 1.0007818937301636
#Bias: 3.9915852546691895
