import cookie
from cookie.model import Model
from cookie.layers import *
from cookie.loss import MSE
from cookie.data import DataLoader
from cookie.optim import SGD


inputs = cookie.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = cookie.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])
data_loader = DataLoader(inputs, labels)
model = Model(layer_list=[Linear(2, 2), Tanh(), Linear(2, 2)])
criterion = MSE()
sgd = SGD(model, learning_rate=0.01)
print(model)
for i in range(1000):
    for batch in data_loader():
        pre = model(batch.feat)
        loss = criterion(prediction=pre, target=batch.label)
        grad = criterion.backward(pre, batch.label)
        sgd.step(grad)
        print (loss)
print (model(inputs).argmax(1), labels.argmax(1))