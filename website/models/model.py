import torch
import pandas as pd
import numpy as np
import syft as sy
import copy
hook = sy.TorchHook(torch)
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, socketio):
        self.model = nn.Linear(8,1)
        self.socketio = socketio

    def trainModel(self, file):
        bob = sy.VirtualWorker(hook, id="bob")
        alice = sy.VirtualWorker(hook, id="alice")
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        bob.add_workers([alice, secure_worker])
        alice.add_workers([bob, secure_worker])
        secure_worker.add_workers([alice, bob])
        self.socketio.emit('message', {'data': 'Looking for diabaetes patients . . . '})
        dataframe=np.genfromtxt(file,delimiter=',',skip_header=1)

        X=dataframe[:,0:8]
        Y=dataframe[:,8]
        Y=Y.reshape(768,1)
        self.socketio.emit('message', {'data': '{} patients found!'.format(X.shape[0])})
        dtype=torch.float
        device=torch.device("cpu")

        from sklearn.preprocessing import StandardScaler
        model=StandardScaler()
        X=model.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        data = torch.from_numpy(X_train)
        target = torch.from_numpy(y_train)
        data,target=data.type(torch.FloatTensor),target.type(torch.FloatTensor)

        data_length,data_width=data.shape

        bobs_data = data[0:int(data_length/2)].send(bob)
        bobs_target = target[0:int(data_length/2)].send(bob)

        alices_data = data[int(data_length/2):].send(alice)
        alices_target = target[int(data_length/2):].send(alice)


        # Initialize A Toy Model

        loss= torch.nn.MSELoss()

        bobs_model = self.model.copy().send(bob)
        alices_model = self.model.copy().send(alice)

        bobs_opt = optim.SGD(params=bobs_model.parameters(),lr=0.001)
        alices_opt = optim.SGD(params=alices_model.parameters(),lr=0.001)

        for i in range(1000):

            # Train Bob's Model
            bobs_opt.zero_grad()
            bobs_pred = bobs_model(bobs_data)
            bobs_loss = loss(bobs_pred,bobs_target)
            bobs_loss.backward()

            bobs_opt.step()
            bobs_loss = bobs_loss.get().data

            # Train Alice's Model
            alices_opt.zero_grad()
            alices_pred = alices_model(alices_data)
            alices_loss = loss(alices_pred,alices_target)
            alices_loss.backward()

            alices_opt.step()
            alices_loss = alices_loss.get().data

        iterations = 10
        worker_iters = 200
        self.socketio.emit('message', {'data': 'Training on two organizations . . .'})
        for a_iter in range(iterations):
            self.socketio.emit('message', {'data': 'Iteration : {}'.format(a_iter+1)})
            bobs_model = self.model.copy().send(bob)
            alices_model = self.model.copy().send(alice)

            bobs_opt = optim.SGD(params=bobs_model.parameters(),lr=0.001)
            alices_opt = optim.SGD(params=alices_model.parameters(),lr=0.001)

            for wi in range(worker_iters):

                # Train Bob's Model
                bobs_opt.zero_grad()
                bobs_pred = bobs_model.forward(bobs_data)
                bobs_loss = loss(bobs_pred,bobs_target)
                bobs_loss.backward()        
                bobs_opt.step()
                bobs_loss = bobs_loss.get().data

                # Train Alice's Model
                alices_opt.zero_grad()
                alices_pred = alices_model.forward(alices_data)
                alices_loss = loss(alices_pred,alices_target)
                alices_loss.backward()
                alices_opt.step()
                alices_loss = alices_loss.get().data
            
            alices_model.move(secure_worker)
            bobs_model.move(secure_worker)
            
            self.model.weight.data.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
            self.model.bias.data.set_(((alices_model.bias.data + bobs_model.bias.data) / 2).get())

        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        X_test,y_test=X_test.type(torch.FloatTensor),y_test.type(torch.FloatTensor)
        pred=self.model(X_test)

        pred=torch.round(pred)
        output=pred-y_test

        incorrect_labels=len(np.nonzero(output))

        accuracy = 1-(incorrect_labels/len(X_test))
        self.socketio.emit('message', {'data': 'Model trained with accuracy: {0:.2f} %'.format(accuracy*100)})

    def testModel(self, data):
        data = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
        pred = torch.round(self.model(data))
        self.socketio.emit('prediction', {'data': 'Prediction: {}'.format(pred)})
