#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


filename = 'ml_err_data.csv'


# In[5]:


df = pd.read_csv(filename,sep=",",header=None,index_col=False)


# In[6]:


# df = df.iloc[1:,:]


# In[7]:


data = np.array(df)


# In[8]:


NDOF = data[:,1]
NDOF = NDOF.flatten()


# In[9]:


Adjoint = data[:,2]
Adjoint = Adjoint.flatten()


# In[10]:


True_Error = data[:,3]
True_Error = True_Error.flatten()


# In[11]:


ML_Error = data[:,4]
ML_Error = ML_Error.flatten()


# In[12]:

plt.figure(1)
plt.loglog(NDOF,Adjoint,"-v",label="Adjoint")
plt.loglog(NDOF,True_Error,"-d",label="True Error")
plt.loglog(NDOF,ML_Error,"-*",label="ML Prediction")
plt.legend()
plt.xlabel("NDOF")
plt.ylabel(r"$|u-u_h|_{L^2(\Omega)}$")
plt.grid(which="both")
plt.savefig("NDOF_Error.png")
# plt.show()


# In[ ]:




