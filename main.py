import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import pandas as pd
from xgboost import XGBRegressor
import numpy.random as nprd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv


# First come up with some function that you would like to try to learn through the two methods.
# can be fun to play around with this and also the domain (lower down in code) to see how this affects the error in learning.



def price(a,d):
    return np.log(a+1) + a**2 -3*a*d - np.log(a*d + 1)

stringfunc = "log(a+1) + a^2 - 3ad - log(ad + 1)"








train_no, test_no, hidden_no, batch_size = 8000, 2000, 100, 100


#This will be our training data
LstX = nprd.randint(0,50,train_no) #generate our x values that we want to train over
LstY = nprd.randint(0,50,train_no) #generate our y values that we want to train over
PriceTrain = [price(x,y) for (x,y) in zip(LstX,LstY)] #create the corresponding output list
N = [[float(LstX[i]),float(LstY[i]), float(PriceTrain[i])] for i in range(train_no)] #put in form for preprocessing
scaler = MinMaxScaler() #this is what we will use for preprocessing
scaler.fit(N) #fit it to our training values
min = scaler.data_min_  #will need these later to invert the transform. These are multivalues atm.
max = scaler.data_max_
LstX1 = [x for [x,_,_] in scaler.fit_transform(N)] #use pattern matching to pick out the scaled xs
LstY1 = [y for [_,y,_] in scaler.fit_transform(N)]
PriceTrain1 = [z for [_,_,z] in scaler.fit_transform(N)]
#df = pd.DataFrame(data = {"x": LstX, "y": LstY, "price": PriceTrain})
df = pd.DataFrame(data = {"x": LstX1, "y": LstY1, "price": PriceTrain1}) #put into a dataframe
#print(df.head(10))


LstT = nprd.randint(0,50,test_no) #very similar process to above but now we're making the testing data.
LstM = nprd.randint(0,50,test_no)
PriceTest = [price(x,y) for (x,y) in zip(LstT,LstM)]
J = [[float(LstT[i]),float(LstM[i]), float(PriceTest[i])] for i in range(test_no)]
LstT1 = [x for [x,_,_] in scaler.fit_transform(J)]
LstM1 = [y for [_,y,_] in scaler.fit_transform(J)]
PriceTest1 = [z for [_,_,z] in scaler.fit_transform(J)]
#df_test = pd.DataFrame(data = {"x": LstT, "y": LstM, "price": PriceTest})
df_test = pd.DataFrame(data = {"x": LstT1, "y": LstM1, "price": PriceTest1})
#print(df_test.head(10))

#Not sure if dataloader likes csv files. So probably easier to define a custom data class.
# Work later could potentially be avoided by defining a collate_fn here as well but
#didn't do this to avoid complications here.
# Found more good info on custom datasets on the pytorch website tutorial.

class Dataclass(Dataset): #data class needs init, len and getitem.
    def __init__(self, data_table):
        super(Dataclass,self).__init__()
        self.data_table = data_table
        self.x = data_table["x"]
        self.y = data_table["y"]
        self.output = data_table["price"]

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, i):
        return list(self.x)[i], list(self.y)[i], list(self.output)[i]




train_set = Dataclass(df) #implement our scaled data as examples of our dataclass.
test_set = Dataclass(df_test)

train_dataloader = DataLoader(train_set,batch_size=batch_size, shuffle = True) #allows us to us dataloader
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle = True) #which has the shuffle command.

#Whip up a neural network

class NeuralNetwork(nn.Module):
    def __init__(self,hidden_layers):
        super(NeuralNetwork,self).__init__()
        self.net = nn.Sequential(nn.Linear(2,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,1)
                                 )

    def forward(self,x):
        run = self.net(x)
        return run

model = NeuralNetwork(hidden_no)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=5e-4) #this learning rate seemed to be quite good for guessing functions.

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y, z) in enumerate(dataloader):
        x = x.view((batch_size,1)) #these steps are what we need to turn the rows into
        y = y.view((batch_size,1)) #a format that can be read by
        Z = z.view((batch_size,1)) #our neural network.
        X = torch.cat((x, y), -1) # --
        X = X.type(torch.float32) # --
        Z = Z.type(torch.float32) # --


        #loss
        pred = model(X)
        loss = loss_fn(pred, Z)

        #backprop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #output losses during training
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for (x, y, z) in dataloader:
            x = x.view((batch_size,1)) #same as above
            y = y.view((batch_size,1))
            Z = z.view((batch_size,1))
            X = torch.cat((x, y), -1)
            X = X.type(torch.float32)
            Z = Z.type(torch.float32)


            pred = model(X)
            test_loss += loss_fn(pred, Z).item()
    test_loss /= num_batches
    p = nprd.randint(0,df_test.shape[0]) #pick a test value to display as sanity check
    (m,n,b) = dataloader.dataset[p]
    V = torch.tensor([[m,n]], dtype=torch.float32)
    B = model(V)
    print(f"Avg loss: {test_loss:>8f} \n")
    print(f"Test run claims: {B}. Actual value: {b}")

epochs = 10 #for learning functions need hundreds/thousands of epochs for reasonably complex functions.
#for t in range(epochs):
#    print(f"Epoch {t + 1}\n-------------------------------")
#    train(train_dataloader, model, loss_fn, optimiser)
#    test(test_dataloader, model, loss_fn)
#print("Training phase complete.")




#torch.save(model.state_dict(), "NN1.pth")  #saves params of nn.
#print("Saved PyTorch Model State") #message to say it was saved.



model = NeuralNetwork(hidden_no) #now implement a neural network of the class neural network evaluate.
model.load_state_dict(torch.load("NN1.pth")) #load the params of the trained network.


#model.eval() #show so evals
#j = nprd.randint(0,df_test.shape[0],size = (10))
#for i in j:
#    x, y, z = df_test["x"][i], df_test["y"][i], df_test["price"][i]
#    with torch.no_grad():
#        X = torch.tensor([[x,y]],dtype = torch.float32)
#        Z = torch.tensor([z], dtype=torch.float32)
#        pred = model(X)
#        predicted, actual = pred, Z
#        print(f'Predicted: "{predicted}", Actual: "{actual}"')




# print("Now implementing XGBoost")
# XGBOOST IMPLEMENTATION

A = pd.concat([df,df_test]) #xgboost likes to split it itself however we could easily keep these seperate.
#print(A.head(10))
A.to_csv('data.csv')


with open('data.csv',newline='') as csvfile:
    results = list(csv.reader(csvfile))

del results[0] #first row was headers
for i in range(len(results)):
    del results[i][0] #first column was indices
Results = np.array(results)

h = Results[:,:2] # first two entries are inputs
j = Results[:,2] # last is output


p = [list(map(float,i)) for i in h]
w = [float(i) for i in j] #different format because of one output vs multiple inputs
X = np.array(p)
Y = np.array(w)

seed = 120 #how it picks a random split for them.
test_size = 0.2 # 80% training data, 20% test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size, random_state=seed)

Model1 = XGBRegressor(objective = 'reg:squarederror', learning_rate = 0.05, n_estimators = 1000)  #https://xgboost.readthedocs.io/en/stable/parameter.html read for different objectives
#Model1.fit(X_train,y_train) #this is the training step.

#score = Model1.score(X_train, y_train) #shows the r2 scores
#print("Training score: ", score)

#ypred = Model1.predict(X_test) #shows the errors
#mse = mean_squared_error(y_test, ypred)
#print("MSE: %.2f" % mse)

#Model1.save_model('model.txt') #if we are using randomly generated datasets for training we want to save
model2 = XGBRegressor() #so we don't get a different xg-model everytime we run code.
model2.load_model('model.txt')




#j = nprd.randint(0,df_test.shape[0],size = (10))
#for i in j:
#    x, y, z = df_test["x"][i], df_test["y"][i], df_test["price"][i]
#    T = np.array([[x,y]])
#    R = model2.predict(T)
#    print(f"Our XGBoost prediction is {R}, and the actual value is {z}")






xmin = min[0] #this is where we split the multivariate min/max from earlier
ymin = min[1]
zmin = min[2]

xmax = max[0]
ymax = max[1]
zmax = max[2]#these are for inverting transform


def NN1(x,y): #turns NN into a classical function so we can plot it.
    model = NeuralNetwork(hidden_no)
    model.load_state_dict(torch.load("NN1.pth"))
    model.eval()
    x1 = (x - xmin)/(xmax - xmin)
    y1 = (y - ymin)/(ymax - ymin)
    Inp = torch.tensor([[x1,y1]], dtype = torch.float32)
    with torch.no_grad():
        pred = model(Inp)
        prediction = pred.item() #because output is only a one tensor we can pick out its value with this.
        Out = prediction*(zmax - zmin) + zmin
    return Out

def xg(x,y): #turns xg into a classical function.
    x1 = (x - xmin) / (xmax - xmin)
    y1 = (y - ymin) / (ymax - ymin)
    Inp = np.array([[x1,y1]])
    Out = model2.predict(Inp)
    Res = Out[0]
    res = Res * (zmax - zmin) + zmin
    return res

def err(f): #a simple error function that should be changed if domain of function changed.
    err = 0
    for i in [i for i in range(0,50,10)]:
        for j in [i for i in range(0,50,10)]:
            err += np.abs(f(i,j) - price(i,j))
    return err

def surf(f): #how we generate the input data for a 3d surface plot based on a function.
    x = [i for i in range(0,50,10)]
    y = [i for i in range(0,50,10)]
    inx = []
    for _ in x:
        for i in x:
            inx.append(i)
    iny = []
    for j in y:
        for _ in y:
            iny.append(j)
    Inx = np.array(inx)
    Iny = np.array(iny)
    out = np.array([[f(i,j)] for (i,j) in zip(inx,iny)])
    Inx = Inx.reshape(5,5)
    Iny = Iny.reshape(5,5)
    out = out.reshape(5,5)
    return Inx, Iny, out

fig = plt.figure() #the above function makes plotting much easier.
fig.suptitle(f"{stringfunc}",fontsize = 16)
ax = fig.add_subplot(221, projection = '3d')
ax.title.set_text("Original")
X,Y,Z = surf(price)
ax.plot_surface(X,Y,Z)


ax2 = fig.add_subplot(222, projection = '3d')
ax2.title.set_text(f"XGBoost with error {round(err(xg),2)}")
X,Y,Z = surf(xg)
ax2.plot_surface(X,Y,Z)

ax3 = fig.add_subplot(223,projection = '3d')
ax3.title.set_text("Original")
X,Y,Z = surf(price)
ax3.plot_surface(X,Y,Z)

ax4 = fig.add_subplot(224,projection = '3d')
ax4.title.set_text(f"NN with error {round(err(NN1),2)}")
X,Y,Z = surf(NN1)
ax4.plot_surface(X,Y,Z)

plt.show()





