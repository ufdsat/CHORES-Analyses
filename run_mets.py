import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, classification_report,confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import losses,optimizers,utils
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import os
import shutil
import glob
import json
import pickle
#tf.compat.v1.disable_eager_execution()
np.random.seed(10)

class MetEstimation:
    
    def __init__(self,task='mets',save_weights=True):
        #pdb.set_trace()
        with open('mets_data.p', 'rb') as fp:
            data = pickle.load(fp)
        self.task = task
        self.make_batches(data)
        if len(glob.glob('metrics/'+self.task+'*.csv'))==0:
            self.initialize_dfs()
       
        self.Flag = save_weights
    
    
    def initialize_dfs(self):
        if not os.path.exists('./history'):
            os.mkdir('history')
        if not os.path.exists('./metrics'):
            os.mkdir('metrics')
        index=['B0','B1','B2','B3','B4','B5','B6','B7','B8','B9']
        cols=['B0','B1','B2','B3','B4','B5','B6','B7','B8','B9']

        self.df= pd.DataFrame(index = index,columns =cols)
        self.df.to_csv('metrics/'+self.task+'_rmse.csv')
            
    def populate_metrics(self):

        file = 'metrics/'+self.task+'_rmse.csv'
        df = pd.read_csv(file,index_col=0)
        df.iat[self.test,self.val]= self.rmse
        df.to_csv(file)
            
    
    def make_batches(self,data):
        #pdb.set_trace()
        keys = data.keys()
        participants = list(keys)
        random.shuffle(participants)
        batches = {}
        X, Y, Z = [], [], []
        for i,participant in enumerate(participants):
            X.extend(data[participant][0])
            Y.extend(data[participant][1])
            Z.append(participant)
            if (i+1)%15==0 or i==len(participants)-1:
                batches[(i-5)//15] = [X,Y,Z]
                X, Y, Z = [], [], []
        self.batches = batches
        self.save_batches()
        
    def save_batches(self):
        keys = self.batches.keys()
        temp = {}
        for key in keys:
            temp[key] = self.batches[key][2]
        filename = self.task+'_batches.json'
        with open(filename, 'w') as outfile:
            json.dump(temp, outfile)
            
    def Model(self):
        inputs = Input(shape=(450,3))
        conv1 = Conv1D(16, 8, strides =1 , padding='same', activation='relu')(inputs)
        conv2 = Conv1D(32, 8, strides =1 , padding='same', activation='relu')(conv1)
        conv3 = Conv1D(64, 8, strides =1 , padding='same', activation='relu')(conv2)
        lstm1 = LSTM(50)(conv3)
        dense = Dense(10,kernel_initializer='normal')(lstm1)
        output = Dense(1,kernel_initializer='normal')(dense)
        model = Model(inputs=inputs,outputs= output)
        return model
    
    
    def plot_Training(self):
        if not os.path.exists('plots/training'):
            os.makedirs('plots/training')
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(self.history.history['loss'],marker='o')
        plt.plot(self.history.history['val_loss'],marker='o')
        plt.grid(True)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig('plots/training/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.png')
        plt.close()
        

    
    def Metrics(self,Y_true,Y_pred):
        self.rmse = sqrt(mean_squared_error(Y_true, Y_pred))
        self.populate_metrics()
        

    def CompilenRun(self,X_train,y_train,val_data):
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))  
        self.model = self.Model()
        self.model.compile(loss=rmse,optimizer='adam')
        earlystop = EarlyStopping(monitor = 'val_loss',patience =5,mode = 'min')
        
        if self.Flag:
            if not os.path.exists('weights/'):
                os.mkdir('weights')
            weights = 'weights/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.h5'
        else:
            weights='weight.h5'
        checkpoint = ModelCheckpoint(weights, monitor='val_loss',save_best_only=True,mode='max')
        self.history = self.model.fit(X_train, y_train,batch_size=256,validation_data=val_data,epochs=50,callbacks=[earlystop,checkpoint])
        self.model.load_weights('weights/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.h5')
        history_dict = self.history.history
        filename = './history/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(history_dict, f)

        
        
        
    def get_data(self,b_id):
        X, y = [] , []
        #pdb.set_trace()
        if isinstance(b_id,np.ndarray):
            for i in b_id:
                X.extend(self.batches[i][0])
                y.extend(self.batches[i][1])
            #X = np.append(X)
        else:
            X = self.batches[b_id][0]
            y = self.batches[b_id][1]
        
        y = np.asarray(y,dtype=np.float32)
        X = np.asarray(X)
        return X,y

    

    
    
    def looprun(self,clean = False):
        
        if clean:
            self.clean()
        batches = np.arange(10)
        for i in range(10):
            self.test = i
            X_test,y_test = self.get_data(self.test)
            train_val = np.delete(batches,i)
            for j in range(len(train_val)):
                K.clear_session()
                self.val = train_val[j]
                self.current = str(self.test)+'_'+str(self.val)
                print(self.current)
                path = 'weights/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.h5'
                if os.path.exists(path):
                    continue
               
                X_val,y_val = self.get_data(self.val)
                #pdb.set_trace()
                train = np.delete(train_val,j)
                X_train,y_train = self.get_data(train)
                self.CompilenRun(X_train,y_train,(X_val,y_val))
                self.plot_Training()
                Y_pred = self.model.predict(X_test,batch_size = 256)
                self.Metrics(y_test,Y_pred)
                K.clear_session()
        self.complete()
        
        
    def complete(self):

        file = 'metrics/'+self.task+'_rmse.csv'
        df = pd.read_csv(file,index_col=0)
        df['mean']=df.mean(axis=1)
        df.to_csv(file)
        
        
        
    def clean(self):
        if os.path.exists('weights'):
            shutil.rmtree('weights')
        if os.path.exists('plots'):
            shutil.rmtree('plots')
        if os.path.exists('metrics/'):
            shutil.rmtree('metrics/')
                
                
 
            
            
            
            
            
            
        

