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
from matplotlib import pyplot as plt
import os
import shutil
import glob
import pdb
import json
import pickle
#tf.compat.v1.disable_eager_execution()
np.random.seed(10)

class ActivityRecognition:
    
    def __init__(self,task,save_weights=False):
        #pdb.set_trace()
        with open('data_new.p', 'rb') as fp:
            data = pickle.load(fp)
        self.task = task
        self.make_batches(data)
        if len(glob.glob('metrics/'+self.task+'*.csv'))==0:
            self.initialize_dfs()
       
        self.Flag = save_weights
    
    
    def initialize_dfs(self):
        if not os.path.exists('./auc'):
            os.mkdir('auc')
        if not os.path.exists('./history'):
            os.mkdir('history')
        if not os.path.exists('./metrics'):
            os.mkdir('metrics')
        columns=['pariticipant','auc','precision','recall','f1_score','balanced_accuracy','specificity','confusion_matrix','support']
        df=pd.DataFrame(columns=columns)
        df.to_csv('metrics/'+self.task+'_comprehensive.csv',index=False)
        index=['B0','B1','B2','B3','B4','B5','B6','B7','B8','B9']
        cols=['B0','B1','B2','B3','B4','B5','B6','B7','B8','B9']
        for column in columns[1:]:
            exec ("self.df_%s= pd.DataFrame(index = index,columns =cols)" % (column))
            exec ("self.df_%s.to_csv('metrics/'+self.task+'_'+column+'.csv')" % (column))
            
    def populate_metrics(self):
        columns = ['auc','precision','recall','f1_score','balanced_accuracy','specificity']
        for column,value in zip(columns,self.metrics[1:-2]):
            file = 'metrics/'+self.task+'_'+column+'.csv'
            df = pd.read_csv(file,index_col=0)
            df.iat[self.test,self.val]= value
            df.to_csv(file)
            
    
    def make_batches(self,data):
        
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
                batches[(i-9)//15] = [X,Y,Z]
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
        conv1 = Conv1D(8, 5, strides =1 , padding='same', activation='relu')(inputs)
        conv2 = Conv1D(16, 5, strides =1 , padding='same', activation='relu')(conv1)
        conv3 = Conv1D(32, 5, strides =1 , padding='same', activation='relu')(conv2)
        conv4 = Conv1D(64, 5, strides =1 , padding='same', activation='relu')(conv3)
        conv5 = Conv1D(128, 5, strides =1 , padding='same', activation='relu')(conv4)
        conv6 = Conv1D(256, 5, strides =1 , padding='same', activation='relu')(conv5)
        lstm1 = LSTM(90)(conv6)
        dense = Dense(10)(lstm1)
        output = Dense(1,activation='sigmoid')(dense)
        model = Model(inputs=inputs,outputs= output)
        return model
        
 
    
    def plot_Training(self):
        if not os.path.exists('plots/training'):
            os.makedirs('plots/training')
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(self.history.history['binary_accuracy'],marker='o')
        plt.plot(self.history.history['val_binary_accuracy'],marker='o')
        plt.grid(True)
        plt.title('accuracy plot')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.subplot(212)
        plt.grid(True)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig('plots/training/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.png')
        plt.close()
        
    def roc_data(self,fpr,tpr):
        df = pd.DataFrame(columns = ['fpr','tpr'])
        df.fpr = fpr; df.tpr = tpr
        df.to_csv('auc/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.csv',index=False)
    
    def Metrics(self,Y_true,Y_pred):
        fpr, tpr, thresholds=roc_curve(Y_true,Y_pred,pos_label=1)
        area = auc(fpr,tpr)
        self.roc_data(fpr,tpr)
        Y_pred[Y_pred>=0.5]=1
        Y_pred[Y_pred<0.5]=0
        classes = ['non-'+self.task,self.task]
        report = classification_report(Y_true,Y_pred,target_names=classes,output_dict=True)
        precision = report[self.task]['precision']
        recall = report[self.task]['recall']
        f1 = report[self.task]['f1-score']
        support = report[self.task]['support']
        specificity =report['non-'+self.task]['recall']
        bacc = balanced_accuracy_score(Y_true, Y_pred)
        conf = confusion_matrix(Y_true, Y_pred)
        print (conf)
        temp = [self.current,area,precision,recall,f1,bacc,specificity,conf,support]
        df = pd.read_csv('metrics/'+self.task+'_comprehensive.csv')
        self.metrics = temp
        df.loc[len(df)]=temp
        df.to_csv('metrics/'+self.task+'_comprehensive.csv',index=False)
        self.populate_metrics()
    
    def CompilenRun(self,X_train,y_train,val_data,train_weights):
        self.model = self.Model()
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
        earlystop = EarlyStopping(monitor = 'val_binary_accuracy',patience =4,mode = 'max')
        
        if self.Flag:
            if not os.path.exists('weights/'):
                os.mkdir('weights')
            weights = 'weights/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.h5'
        else:
            weights='weight.h5'
        checkpoint = ModelCheckpoint(weights, monitor='binary_accuracy',save_best_only=True,mode='max')
        self.history = self.model.fit(X_train, y_train,batch_size=256,validation_data=val_data,class_weight=train_weights,epochs=50,callbacks=[earlystop,checkpoint])
        self.model.load_weights('weights/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.h5')
        history_dict = self.history.history
        filename = './history/'+self.task+'_'+str(self.test)+'_'+str(self.val)+'.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(history_dict, f)

        
        
        
    def get_data(self,b_id,sampling=False):
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
        
        y = np.asarray(y,dtype=np.str)
        X = np.asarray(X)
        if sampling:
            X,y = self.filter_val(X,y)
        y = np.where(y==self.task,1,0)
        return X,y

    
    ############################Deprecated#####################################
    def filter_val_deprecated(self,X,y):
        one_inds = np.array(np.where(y==self.task)).squeeze()
        zero_inds = np.array(np.where(y!=self.task)).squeeze()
        if len(one_inds)>len(zero_inds):
            one_inds, zero_inds = zero_inds, one_inds           
        X = np.array(X)
        inds = np.random.choice(zero_inds,len(one_inds),replace=False)
        inds = np.concatenate([one_inds,inds])
        X = X[inds]; y= y[inds]
        return X,y 
    ############################################################################
    
    def filter_val(self,X,y):
        #pdb.set_trace()
        tasks = ['sedentary','locomotion','lifestyle']
        tasks.remove(self.task)
        zero_inds = np.array(np.where(y==self.task)).squeeze()
        one_inds = np.array(np.where(y==tasks[0])).squeeze()
        two_inds = np.array(np.where(y==tasks[1])).squeeze()
        if len(zero_inds)>len(one_inds)+len(two_inds):
            one_inds, zero_inds = zero_inds, one_inds  
        
        X = np.array(X)
        inds1 = np.random.choice(one_inds,len(zero_inds) if len(zero_inds) <= len(one_inds) else len(one_inds),replace=False)
        inds2 = np.random.choice(two_inds,len(zero_inds) if len(zero_inds) <= len(two_inds) else len(two_inds),replace=False)
        inds = np.concatenate([zero_inds,inds1,inds2])
        X = X[inds]; y= y[inds]
        return X,y 
    
    
    def looprun(self,clean = False):
        #pdb.set_trace()
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
               
                X_val,y_val = self.get_data(self.val,sampling=True)
                train = np.delete(train_val,j)
                X_train,y_train = self.get_data(train)
                train_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
                self.CompilenRun(X_train,y_train,(X_val,y_val),train_weights)
                self.plot_Training()
                Y_pred = self.model.predict(X_test,batch_size = 256)
                self.Metrics(y_test,Y_pred)
                K.clear_session()
        self.complete()
        
        
    def complete(self):
        columns = ['auc','precision','recall','f1-score','balanced_accuracy']
        for column in columns:
            file = 'metric/'+column+'.csv'
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
                
                
 
            
            
            
            
            
            
        

