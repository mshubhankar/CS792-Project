import torch
import pandas as pd
import numpy as np
import syft as sy
import copy
import argparse
hook = sy.TorchHook(torch)
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self, socketio,partition):
        self.model = nn.Linear(8,1)
        self.scaler = StandardScaler()
        self.socketio = socketio
        self.partition = partition

    def trainModel(self, file):
        bob = sy.VirtualWorker(hook, id="bob")
        alice = sy.VirtualWorker(hook, id="alice")
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")
        partition = self.partition
        print(partition)
        bob.add_workers([alice, secure_worker])
        alice.add_workers([bob, secure_worker])
        secure_worker.add_workers([alice, bob])
        self.socketio.emit('message', {'data': 'Looking for diabetes patients . . . '})
        dataframe=np.genfromtxt(file,delimiter=',',skip_header=1)

        X=dataframe[:,0:8]
        Y=dataframe[:,8]
        Y=Y.reshape(768,1)
        dtype=torch.float
        device=torch.device("cpu")

        X=self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=560)

        data = torch.from_numpy(X_train)
        target = torch.from_numpy(y_train)
        data,target=data.type(torch.FloatTensor),target.type(torch.FloatTensor)

        data_length,data_width=data.shape

        bobs_data = data[0:int(data_length*partition)].send(bob)
        self.socketio.emit('message', {'data': '{} patients found in Organization A'.format(bobs_data.shape[0])})
        bobs_target = target[0:int(data_length*partition)].send(bob)

        alices_data = data[int(data_length*(partition)):].send(alice)
        self.socketio.emit('message', {'data': '{} patients found in Organization B'.format(alices_data.shape[0])})
        alices_target = target[int(data_length*(partition)):].send(alice)

        self.socketio.emit('message', {'data': 'Total patients found : {}'.format(bobs_data.shape[0]+alices_data.shape[0])})

        self.socketio.emit('send_model')

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

        self.socketio.emit('start_model');

        iterations = 10
        worker_iters = 200
        a_progress = 0
        b_progress = 0
        for a_iter in range(iterations):
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
                if not (wi % 20):
                    a_progress += 1
                    self.socketio.emit('bar1_update', {'data' : '{}%'.format(a_progress)});

                # Train Alice's Model
                alices_opt.zero_grad()
                alices_pred = alices_model.forward(alices_data)
                alices_loss = loss(alices_pred,alices_target)
                alices_loss.backward()
                alices_opt.step()
                alices_loss = alices_loss.get().data
                if not (wi % 20):
                    b_progress += 1
                    self.socketio.emit('bar2_update', {'data' : '{}%'.format(b_progress)});
            alices_model.move(secure_worker)
            bobs_model.move(secure_worker)
            
            self.model.weight.data.set_(((alices_model.weight.data*(1-partition) + bobs_model.weight.data*partition)).get())
            self.model.bias.data.set_(((alices_model.bias.data*(1-partition) + bobs_model.bias.data*partition) ).get())

        self.socketio.emit('stop_model')

        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        X_test,y_test=X_test.type(torch.FloatTensor),y_test.type(torch.FloatTensor)
        pred=self.model(X_test)

        pred=torch.round(pred)
        output=pred-y_test

        incorrect_labels=len(np.nonzero(output))

        accuracy = 1-(incorrect_labels/len(X_test))
        from sklearn.metrics import classification_report #Testing Report
        testing_report = classification_report(y_test,pred.detach().numpy())
        # print(testing_report)
        # testing_report = testing_report.replace('\n', '<br> <br />')
        # self.socketio.emit('testing_report', {'data': 'External Validation Report:\n {} \n'.format(testing_report)})
        # self.socketio.emit('testing_accuracy', {'data': 'Testing accuracy: {0:.2f} % \n'.format(accuracy*100)})

        X_train=torch.from_numpy(X_train)
        y_train=torch.from_numpy(y_train)
        X_train,y_train=X_train.type(torch.FloatTensor),y_train.type(torch.FloatTensor)
        pred=self.model(X_train)
        pred=torch.round(pred)

        training_report = classification_report(y_train,pred.detach().numpy())
        output=pred-y_train
        incorrect_labels=len(np.nonzero(output))

        training_accuracy = 1-(incorrect_labels/len(X_train))
        # training_report = training_report.replace('\n', '<br />')

        # self.socketio.emit('training_report', {'data': 'Internal Validation Report:\n {} '.format(training_report)})
        self.socketio.emit('training_accuracy', {'data0': 'Training accuracy: {0:.2f} %'.format(training_accuracy*100),
            'data1': 'External Validation Report:\n {} \n'.format(testing_report),
            'data2': 'Testing accuracy: {0:.2f} % \n'.format(accuracy*100),
            'data3': 'Internal Validation Report:\n {} '.format(training_report)})
            

    def testModel(self, data):
        data = torch.from_numpy(self.scaler.transform(np.array([data]))).type(torch.FloatTensor)
        pred = torch.round(self.model(data))
        self.socketio.emit('prediction', {'data': 0 if pred == 0 else 1},
            'testing_report', {'data': 'External Validation Report:\n {} \n'.format(testing_report)})
