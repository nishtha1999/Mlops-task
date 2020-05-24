#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


train, test = dataset


# In[4]:


X_train, y_train = train
X_test, y_test = test


# In[5]:


X_train_1d = X_train.reshape(-1, 28*28)
X_test_1d = X_test.reshape(-1, 28*28)


# In[6]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[7]:


from keras.utils.np_utils import to_categorical


# In[8]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[9]:


from keras.models import Sequential


# In[10]:


from keras.layers import Dense


# In[11]:


model = Sequential()


# In[12]:


model.add(Dense(units=512,activation='relu', input_dim=28*28))


# In[13]:


model.add(Dense(units=256,activation='relu'))


# In[14]:


model.add(Dense(units=128,activation='relu'))


# In[15]:


model.add(Dense(units=64,activation='relu'))


# In[16]:


model.add(Dense(units=10,activation='softmax'))   #softmax for multi-classification


# In[17]:


model.summary()


# In[18]:


from keras.optimizers import Adam


# In[19]:


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[20]:


model.fit(X_train, y_train_cat, validation_data=(X_test,y_test_cat), epochs=4)


# In[21]:


accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print("%.2f%%" % (accuracy[1]*100))


# In[22]:


accuracy = accuracy[1]*100


# In[31]:


accuracy = str(accuracy)


# In[32]:


accuracy


# In[34]:


import smtplib
server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login("nishthasharma672@gmail.com","#yourpassword")
msg = accuracy
server.sendmail("nishthasharma672@gmail.com","nishthasharma006@gmail.com",msg)
server.quit()


# In[ ]:




