#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler


# In[2]:

#sigmoid function for estimation of predicted value
def sigmoid(z):
  return 1/(1+np.exp(-z))


# In[6]:


def costfunction(beta, x, y): 
  epsilon = 0.00000000000000000000000000001
  y_hat = sigmoid(np.dot(x, beta))
  c_y = y*np.log(y_hat+epsilon)
  c_yi = (1-y)*np.log(1-y_hat+epsilon)
  cost_function = -sum(c_y+c_yi)/len(x)
  return cost_function


# In[7]:


def logFit(x, y, beta):
  learning_rate = 0.1
  cost_func_list = [100000000, 10000000]
  loss = []
  while x.shape[1] > 0:
    y_hat = sigmoid(np.dot(x, beta))
    x_u = (learning_rate*np.subtract(y_hat, y))/len(x)
    beta -= np.dot(x.transpose(), x_u)
  #cost_function:
    cost_func = costfunction(beta, x, y)
    cost_func_list.append(cost_func)
    loss.append(cost_func)
    if cost_func_list[-1]>cost_func_list[-2]:
      break
    if cost_func_list[-1] == 0:
      break
    if cost_func_list[-1] < 0:
      break
  return beta, loss


# In[8]:


def logPredict(x, beta, threshold):
  y_hat_new = np.dot(x, beta)
  y_predict = sigmoid(y_hat_new)
  for i in range(len(y_predict)):
    if y_predict[i] >= threshold:
      y_predict[i] = 1    
    else:
      y_predict[i] = 0
  return y_predict


# In[9]:


#building confusion matrix
def confusion(y,y_hat):
  import pandas as pd
  df = pd.DataFrame(y)
  df["y_hat"] = y_hat
  df.columns = ["actual", "predicted"]
  tp = len(df[(df["actual"]==1)&(df["predicted"]==1)])
  fp = len(df[(df["actual"]==0)&(df["predicted"]==1)])
  fn = len(df[(df["actual"]==1)&(df["predicted"]==0)])
  tn = len(df[(df["actual"]==0)&(df["predicted"]==0)])
  return tp, fp, fn, tn

def confusionMatrix(y, y_hat):
  tp, fp, fn, tn = confusion(y,y_hat)
  matrix = pd.DataFrame([[tp, fp],[fn, tn]])
  matrix.columns = ["Actual Positive,", "Actual Negative"]
  matrix.index = ["Predicted Positive", "Predicted Negative"]
  return matrix


# In[10]:


#accuracy calculation
def precision(y,y_hat):
  tp, fp, fn, tn = confusion(y,y_hat)
  prec = tp/(tp+fp)
  return prec

def recall(y,y_hat):
  tp, fp, fn, tn = confusion(y,y_hat)
  rec = tp/(tp+fn)
  return rec

def f1Score(y,y_hat):
  preci = precision(y,y_hat)
  reca = recall(y,y_hat)
  f_score = (2 * preci * reca)/(preci+reca)
  return f_score
