import random 

target : float = 2
prediction : float = 0

weight : float = random.random()
target : float = float(input("What should the product be?"))
inputx : float = float(input("What number to start with, please not to large?"))
bias : float = 0
learning_rate : float = float(input("Set learning rate."))

def lossf(prediction):
    result : float = 2
    loss : float = (abs(result - prediction))**2 #MSE
    return loss

for i in range(1000):
    loss =lossf(prediction) 
    prediction = inputx * weight + bias
    dw = 2 * (prediction - target) * inputx #need to look into more
    weight = weight - learning_rate * dw
    if loss == 0:
        break


print(prediction)
print(weight)