#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 21 04:07:50 2020
@author: c1ph3r
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Input, Dense
from keras.layers.core import Lambda
from keras import backend as K
from keras.utils import plot_model
from tweek import set_size
#for saving models
from keras.models import model_from_yaml
from keras.models import model_from_json
#plotting modules
#import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#plt.style.use('mpsty/mypaperstyle.mplstyle')
#mpl.use('pdf')
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('seaborn-paper')
'''plt.rc('font', family= 'serif', serif= 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman')
#plt.rc('font', family='sans-serif', sans_serif= 'Helvetica, Avant Garde, Computer Modern Sans serif')
plt.rc('font', family='sans-serif')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('font', family='monospace', monospace= 'Courier, Computer Modern Typewriter')'''
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex = True) # Use latex for text
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 14)
plt.rc('axes', linewidth=1)
plt.rcParams['text.latex.preamble'] =[r'\boldmath']
params = {'legend.framealpha': 0.1,
          'legend.handlelength': 0.8,
          'legend.labelspacing':0.1,
          'legend.fontsize' : 10}
plt.rcParams.update(params)


class Ofa:
    def __init__(self,
                X,y,F,ts, q=0.05, batch_size=250, epochs=30, ly_one=128, ly_two=32, ext=True):
        self.ext = ext
        self.split_ratio = ts
        self.X, self.y,self.F = X, y,F
        self.sequence_length = self.X.shape[1]
        self.ts = int(self.split_ratio* self.X.shape[0]) #for hrss 0.7495
        self.nl_1=ly_one
        self.nl_2=ly_two
        self.batch_size = batch_size
        self.epochs = epochs
        self.q=q
        self.y_train = self.y[0:self.ts]

    def fit(self, dropout=0.3, embedd=True):
        # fix random seed for reproducibility
        np.random.seed(7)
        self.input_ae = Input(shape=(self.sequence_length, self.X.shape[2]))
        self.lstm1 = LSTM(self.nl_1, return_sequences=True, dropout=dropout)
        encoded_ae = self.lstm1(self.input_ae, training=True)
        self.encoder = Model(self.input_ae, encoded_ae)
        self.encoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])
        self.Xe = self.encoder.predict(self.X) # generate X feature vectors
        #self.ext=ext
        self.embedd=embedd
        if self.ext: #If exegenous features are available
            if embedd:#FXe
                dim1=self.F.shape[2]
                dim2=self.nl_1
                self.FXe = np.concatenate([self.Xe, self.F], axis=2) #concatenate X, F
                self.XFe_train = self.FXe[0:self.ts,:]
                self.XFe_test = self.FXe[self.ts:,:]
                self.hxfe, model, scaler= self.train(self.XFe_train, self.XFe_test, dim1,dim2)
                self.x_test=self.X[self.ts:,:]
                print('..Fitting exogenous and embedding vectors (XFe) done..')

            else:#FX ex with no embedding
                dim1=self.F.shape[2]
                dim2=self.X.shape[2]
                self.FX = np.concatenate((self.F, self.X), axis=2)
                self.FX_train = self.FX[0:self.ts,:]
                self.FX_test = self.FX[self.ts:,:]
                self.hxf, model, scaler=self.train(self.FX_train, self.FX_test, dim1, dim2)
                self.x_test=self.X[self.ts:,:]
                print('...Fitting features (FX) done...')

        else:#No exogenous features
            if embedd:#Xe with embedding
                dim1=0
                dim2=self.nl_1
                self.Xe_train = self.Xe[0:self.ts,:]
                self.Xe_test= self.Xe[self.ts:,:]
                self.hxe, model, scaler=self.train(self.Xe_train, self.Xe_test, dim1, dim2)
                self.x_test=self.X[self.ts:,:]
                print('.. Fitting embending vectors (Xe) done..')

            else:#No ex and no embedding
                dim1= 0
                dim2=self.X.shape[2]
                self.X_train = self.X[0:self.ts,:]
                self.X_test = self.X[self.ts:,:]
                self.hx, model, scaler = self.train(self.X_train, self.X_test, dim1, dim2)
                #self.x_test=self.X[self.ts:,:]
                print('..fitting features (X) done..')
        self.model, self.scaler, self.dim1, self.dim2 = model, scaler, dim1, dim2
        #return self.model, self.scaler,self.dim1, self.dim2

    def train(self, Xtr,Xte,di1,dim2):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr.reshape(-1,dim2+di1)).reshape(-1,self.sequence_length,dim2+di1)
        Xte = scaler.transform(Xte.reshape(-1,dim2+di1)).reshape(-1,self.sequence_length,dim2+di1)
        #self.XF_val_e = self.scaler1.transform(self.XF_val_e.reshape(-1,self.nl_1+dim)).reshape(-1,self.sequence_length,self.nl_1+dim)
        inputs = Input(shape=(Xtr.shape[1], Xtr.shape[2]))
        lstm1 = LSTM(self.nl_1, return_sequences=True, dropout=0.3)(inputs, training=True)
        lstm2 = LSTM(self.nl_2, return_sequences=False, dropout=0.3)(lstm1, training=True)
        dense1 = Dense(50)(lstm2)
        out10 = Dense(1)(dense1)
        out50 = Dense(1)(dense1)
        out90 = Dense(1)(dense1)
        self.losses = [lambda y,f:self.loss(self.q, y, f), lambda y,f:self.loss(0.5, y, f), lambda y,f: self.loss(1-self.q, y, f)]
        #out = Dense(1)(dense1)
        model = Model(inputs, [out10,out50,out90])
        model.compile(loss=self.losses, optimizer='adam', metrics=['mae'], loss_weights = [0.2,0.2,0.2])
        #model = Model(inputs, out)
        #model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
        history = model.fit(Xtr, [self.y_train, self.y_train, self.y_train], \
                                            epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=True)

        '''self.history1 = self.model1.fit(self.XF_train_e, self.y_train, \
                                        validation_data=(self.XF_val_e,self.y_val),epochs=150, batch_size=128, verbose=1, shuffle=True)'''
        return history, model, scaler

    '''def evaluate(self, choice='True'):
        if choice:
            print('validating......', end='\n')
            self.val_score = self.model.fit(self.XF_val_noe, self.y_val, epochs=30, batch_size=128, verbose=1, shuffle=True)
        else:
            print('Testing......', end='\n')
            self.test_score = self.model.fit(self.XF_test_noe, self.y_test, epochs=30, batch_size=128, verbose=1, shuffle=True)'''

    def predictor(self, ext, emb):
        #self.model, self.scaler,self.dim1, self.dim2 =self._fit()
        if emb:
            if ext:
                enc = K.function([self.encoder.layers[0].input, K.learning_phase()], [self.encoder.layers[-1].output])
                NN = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])
                enc_pred = np.vstack(enc([self.x_test,1]))
                enc_pred = np.concatenate([enc_pred, self.F[self.ts:,:]], axis=2)
                trans_pred = self.scaler.transform(enc_pred.reshape(-1,self.dim1+self.dim2)).reshape(-1,self.sequence_length,self.dim1+self.dim2)
                NN_pred = NN([trans_pred,1])
                #print('...quantile applied on XFe done...')
            else:
                enc = K.function([self.encoder.layers[0].input, K.learning_phase()], [self.encoder.layers[-1].output])
                NN = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])
                enc_pred = np.vstack(enc([self.x_test,1]))
                #enc_pred = np.concatenate([enc_pred, self.F[self.ts:,:]], axis=2)
                trans_pred = self.scaler.transform(enc_pred.reshape(-1,self.dim2)).reshape(-1,self.sequence_length,self.dim2)
                NN_pred = NN([trans_pred,1])
                #print('...quantile applied on Xe done...')

        else:
            if ext:
                NN = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])
                #self.XF_train_noe = np.concatenate((self.x_test, self.F_test), axis=2)
                trans_pred = self.scaler.transform(self.FX[self.ts:,:].reshape(-1,self.dim2+ \
                                                                               self.dim1)).reshape(-1,self.sequence_length,self.dim2+self.dim1)
                NN_pred = NN([trans_pred,1])
                #print('..quantile applied on XF done..')

            else:
                NN = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])
                #self.XF_train_noe = np.concatenate((self.x_test, self.F_test), axis=2)
                trans_pred = self.scaler.transform(self.X[self.ts:,:].reshape(-1,self.dim2)).reshape(-1,self.sequence_length,self.dim2)
                NN_pred = NN([trans_pred,1])
                #print('..quantile appliedt on X done..')

        return NN_pred

    def plot(self,name='testing'):
        #plt.figure(figsize=(16,8))
        #plt.xlabel(r'$\bf {running time(s)} x10$', fontsize=45, fontweight ='bold')
        #plt.ylabel(r'$\bf {window size~}x100$', fontsize=45, fontweight ='bold')
        #plt.xlabel(r'\textbf{execution time(s)} $x10$', fontsize=10, fontweight ='bold')
        #plt.ylabel(r'$\psi~x10^2$', fontsize=10, fontweight ='bold')
        fraction = 0.5
        width = 512
        fig, ax = plt.subplots(3,figsize=set_size(width, fraction), sharex='all',sharey='all', gridspec_kw={'hspace': 0.5})
        ax[0].scatter(range(0,len(self.y_test)),self.y_test, s=50, marker='*', c='blue', alpha=0.7, label='test data')
        ax[1].plot(self.error, c='green', label='scores', alpha=0.9, lw=2)
        for i in range(2):
            #ax[i].xaxis.set_major_locator(plt.LinearLocator(6))
            #ax[i].yaxis.set_major_locator(plt.LinearLocator(3))
            #ax[i].xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            #ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax[i].yaxis.set_minor_locator(AutoMinorLocator(2))
            #ax[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        #plot anomalies
        plt.scatter(np.where(np.logical_or(self.y_test>self.y90, self.y_test<self.y10))[0],
                    self.anom_scores, c='red', s=50, marker='*',  alpha=0.7)
        plt.plot(self.error, c='green', alpha=0.9, lw=2)
        fig.text(-0.005, 0.5, 'OFA scores', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
        plt.xlabel(r'time')
        #plt.ylabel(r'OFA score')
        plt.tight_layout()
        # plt.savefig('/media/c1ph3r/colls/Dropbox/_China/_Xidian/_6th/vldb/vldb_style_sample/latex/figures/'+name+'.pdf',
        #             format='pdf', bbox_inches='tight')

        fraction = 0.5
        width =510
        fig, ax = plt.subplots(3,figsize=set_size(width, fraction), sharex='all', gridspec_kw={'hspace': 0.5})
        ax[0].plot(self.y90, c='orange', alpha=0.9, lw=2)
        ax[1].plot(self.y50, c='cyan', alpha=0.9, lw=2)
        ax[2].plot(self.y10, c='purple', alpha=0.9, lw=2)
        for i in range(3):
            #ax[i].xaxis.set_major_locator(plt.LinearLocator(6))
            ax[i].yaxis.set_major_locator(plt.LinearLocator(3))
           # ax[i].xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
            #ax[i].yaxis.set_minor_locator(AutoMinorLocator(1))
            #ax[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        plt.xlabel(r'time')
        #plt.ylim([-0.001, 0.001])
        plt.tight_layout()
        #fig.text(0.1, 0.5, 'scores', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
        # plt.savefig('/latex/figures/credit.pdf',
        #             format='pdf',bbox_inches='tight')

    def predict(self, X,y,F):
        self.x_test, self.y_test, self.f_exogenous = X,y,F
        #self.x_test, self.y_test = X,y
        #tracking from the encoded states
        x_enc = self.encoder.predict(self.x_test)
        xf=np.concatenate((x_enc, self.f_exogenous), axis=2)
        self.yhat = np.array(self.model.predict(xf))[:,]
        #using predictor
        self.scores= np.array(self.predictor(1,self.ext, self.embedd)) #lower, median and upper quantile scores
        self.y10, self.y50, self.y90 = self.scores[0][:,0], self.scores[1][:,0], self.scores[2][:,0]
        self.mean_s, self.std_s= np.mean(self.y50.mean(axis=0), axis=0), np.std(self.y50.mean(axis=0), axis=0)
        self.anomaly_scores()
        self.error= self.y_test-self.y50

    def _predict(self, name='stock'):
        fraction = 0.5
        width =510
        fig, ax = plt.subplots(3, figsize=set_size(width, fraction), sharex='all', gridspec_kw={'hspace': 0.5})
        ax[0].plot(self.y10[28:], color='green', alpha=0.9)
        ax[0].plot(self.y50[28:], color='blue', alpha=0.6)
        ax[0].plot(self.y90[28:], color='purple', alpha=0.8)
        ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax[0].yaxis.set_major_locator(plt.LinearLocator(3))
        ax[0].set_title('fitted model (blue) and thresholds')
        #ax[0].setylabel(r'thresholds')
        ax[1].plot(self.y_test[28:], alpha=0.9, lw=2, c='orange')
        ax[1].yaxis.set_major_locator(plt.LinearLocator(3))
        ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax[1].set_title('test data')
        ax[2].plot(self.anom_scores[28:], c='navy', label='scores', alpha=0.9, lw=2)
        ax[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax[2].yaxis.set_major_locator(plt.LinearLocator(3))
        #ax[2].plot(self.id_anomaly)
        ax[2].scatter(np.where(np.logical_or(self.y_test[28:]>self.y90[28:], self.y_test[28:]<self.y10[28:]))[0],
                     self.anom_scores[28:][np.logical_or(self.y_test[28:]>self.y90[28:], self.y_test[28:]<self.y10[28:])], c='red', s=50, marker='*',  alpha=0.7)
        # ax[2].scatter(np.where(np.logical_or(self.y_test[28:]>self.y90[28:], self.y_test[28:]<self.y10[28:]))[0],
        #             self.anom_scores[28:][self.id_anomaly], c='red', s=50, marker='*',  alpha=0.7) #best simplified version for marking anomalies
        plt.ylim([-0.27,0.1])
        ax[2].xaxis.set_major_locator(plt.LinearLocator(6))
        fig.text(-0.06, 0.5, 'OFA scores', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
        plt.xlabel(r'time', fontsize=plt.rcParams['axes.labelsize'])
        plt.tight_layout()

        #plot anomalies

        #ax.xaxis.set_major_locator(plt.LinearLocator(6))
        #ax.yaxis.set_major_locator(plt.LinearLocator(4))
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        #ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        #ax.yaxis.set_major_formatter(ScalarFormatter())
        #ax.yaxis.major.formatter._useMathText = True
        #ax.yaxis.major.formatter._useMathText = True
        #ax.yaxis.set_minor_locator(AutoMinorLocator(1))
        #plt.ylim([self.y10.min()-1, self.y90.max()+1])
        #plt.legend(loc='lower right')
        #plt.tight_layout()

        #fig, ax = plt.subplots(figsize=set_size(width, fraction))
        #ax[1].plot(self.error, c='green', label='scores', alpha=0.9, lw=2)
        plt.savefig('/latex/figures/'+
                     name+'.pdf',format='pdf',bbox_inches='tight')


    def gen_sequence(self,df, seq_length, seq_cols):
        data = df[seq_cols].values
        num_elements = data.shape[0]

        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data[start:stop, :]

    def read_dataset(self, filename, targX, sequence_length, ext=True):
        data = pd.read_csv(filename, sep=',', usecols=[3,4,5,6,7])
        #filename = './datasets/physics.dat'
        '''r = open(filename,'r').read().split(',')
        X = np.array(list(map(float,r)))
        data = pd.DataFrame({'LOW':X})'''
        #filename='datasets/high-storage-system-data-for-energy-optimization/HRSS_anomalous_optimized.csv'
        #data = pd.read_csv(filename, sep=',', usecols=[0,1,2,3,6],names=['temp','hum','pre','mea','LOW'])
        df = data
        X,F = [], []
        x = data[targX].values
        y = x[sequence_length:x.shape[0]]
        for sequence in self.gen_sequence(df, sequence_length, [targX]):
            X.append(sequence)
        X = np.array(X)

        if ext:
            cols = [cols for cols in data.columns if cols !=targX]
            for sequence in self.gen_sequence(df, sequence_length, cols):
                F.append(sequence)
        F = np.array(F)
        return  X, F, y

        def read_data(self, data,sequence_length, ext=False):
            #filename = 'datasets/edf_stocks.csv'
            #data = pd.read_csv(filename, sep=',', usecols=[3,4,5,6,7])
            #data = pd.read_csv(filename, sep=',', usecols=[0,1,2,3,6],names=['temp','hum','pre','mea','LOW'])
            dfxy = pd.DataFrame(data)
            del dfxy[0]
            X, F = [], []
            targX = [cols for cols in dfxy.columns]
            xy = dfxy[targX].values
            y = xy[sequence_length:xy.shape[0],-1]
            for sequence in self.gen_sequence(dfxy, sequence_length, targX):
                X.append(sequence)
            X = np.array(X)

            if ext:
                cols = [cols for cols in dfxy.columns if cols !=targX]
                for sequence in self.gen_sequence(dfxy, sequence_length, cols):
                    F.append(sequence)
            F = np.array(F)
            return X, F, y

    def anomaly_scores(self):
        #return index of anomalies in the test data
        self.anom_scores = self.y_test -self.y50
        self.anomaly = self.y_test[np.logical_or(self.y_test>self.y90, self.y_test<self.y10)]
        ### CROSSOVER CHECK ###
        id_anomaly=[]
        for i,v in enumerate(self.y_test):
                if np.logical_or(self.y_test[i]>self.y90[i], self.y_test[i]<self.y10[i]):
                    id_anomaly.append(i)
        self.id_anomaly = np.array(id_anomaly)


    def loss(self, q,y,f):
        e = y-f
        return K.mean(K.maximum(q*e, (q-1)*e),axis=-1)

    def modelsavn(self,savejs=False, saveym=False, loadym=False, loadjs=False, savemodel=True):
        #needs further processessing
        # serialize model to JSON
        if savejs:
            self.model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(self.model_json)
            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            print("Saved model to disk")

        # later...
        if loadjs:
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            print("Loaded model from disk")

        # serialize model to YAML
        if saveym:
            model_yaml = self.model.to_yaml()
            with open("model.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml)
            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            print("Saved model to disk")
        #Later
        if loadym:
            yaml_file = open('model.yaml', 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            loaded_model = model_from_yaml(loaded_model_yaml)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            print("Loaded model from disk")

        if savemodel:
            # save model and architecture to single file
            self.model.save("model.h5")
            print("Saved model to disk")

        return loaded_model
        ''' evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))'''








