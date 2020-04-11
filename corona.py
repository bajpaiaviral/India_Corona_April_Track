#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:08:19 2020

@author: aviral
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

america_csv=pd.read_csv('us_covid19_daily.csv')
america=america_csv.sort_index(axis=1 ,ascending=True)
america = america.iloc[::-1]
america_cases=america['positive']

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

america_input,america_output=split_sequence(america_cases,5)
america_output=america_output[::-1]
america_output=america_output[5:]
america_output=np.append(america_output,[333220])
america_output=np.append(america_output,[362171])
america_output=np.append(america_output,[393357])
america_output=np.append(america_output,[423637])
america_output=np.append(america_output,[457963])
america_cases=america_cases[::-1]
america_cases.reindex(index=america_cases.index[::-1])

america_cases.plot(kind='bar')

# define model
n_steps = 5
n_features = 1
X=america_input
print(X.shape)
X = X.reshape((X.shape[0], X.shape[1], n_features))




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, america_output, epochs=550, verbose=1)
print(model.history)
print(X)
x=np.array([[3500,3900,4300,4800,5499]])
x = x.reshape((x.shape[0], x.shape[1], n_features))

print(model.predict(x))
pd.DataFrame(america_output).plot(kind='bar')


indiacorona=pd.read_csv('covid_19_india.csv')

indiacorona["Date"] = pd.to_datetime(indiacorona["Date"])
indiacorona=indiacorona.sort_values(by="Date")
india=indiacorona.groupby(['Date']).sum()

indiastart=india[india['Confirmed'] >=1834]
indiastart=indiastart['Confirmed']

inputlist=[]
inputlist.append(4281)
inputlist.append(4789)
inputlist.append(5274)
inputlist.append(5865)
inputlist.append(6761)
output=np.array(indiastart)

for i in range(20):
    x=np.array([[inputlist[0],inputlist[1],inputlist[2],inputlist[3],inputlist[4]]])

    x = x.reshape((x.shape[0], x.shape[1], n_features))

    y=model.predict(x)
    inputlist.remove(inputlist[0])
    inputlist.append(y)
    output=np.append(output,[int(y)])
    
pd.DataFrame(output).plot(kind='bar')
   

deathsamerica=america['death']
damerica_input,damerica_output=split_sequence(deathsamerica,5)
damerica_output=damerica_output[::-1]
damerica_output=damerica_output[5:]
damerica_output=np.append(damerica_output,[9542])
damerica_output=np.append(damerica_output,[10705])
damerica_output=np.append(damerica_output,[12628])
damerica_output=np.append(damerica_output,[14495])
damerica_output=np.append(damerica_output,[16399])


n_steps = 5
n_features = 1
X=damerica_input
print(X.shape)
X = X.reshape((X.shape[0], X.shape[1], n_features))




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, damerica_output, epochs=550, verbose=1)
print(model.history)

dindia=india[india['Deaths'] >=41]
dindia=dindia['Deaths']
    
inputlist=[]
inputlist.append(111)
inputlist.append(124)
inputlist.append(149)
inputlist.append(169)
inputlist.append(206)
doutput=np.array(dindia)

for i in range(20):
    x=np.array([[inputlist[0],inputlist[1],inputlist[2],inputlist[3],inputlist[4]]])

    x = x.reshape((x.shape[0], x.shape[1], n_features))

    y=model.predict(x)
    inputlist.remove(inputlist[0])
    inputlist.append(y)
    doutput=np.append(doutput,[int(y)])
    
pd.DataFrame(doutput).plot(kind='bar')
pd.DataFrame(output).plot(kind="bar")
output.index=output.index+1
pd.DataFrame(output).plot(kind="bar",title="Cases Projected in April")
pd.DataFrame(doutput).plot(kind='bar',title="Deaths Projected in April")



# split a univariate sequence into samples
