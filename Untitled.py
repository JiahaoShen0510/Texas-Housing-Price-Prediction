#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[2]:


data=pd.read_csv("Texas Housing.csv")
data


# In[3]:


data.describe()


# In[4]:


data.isnull().sum()


# In[5]:


data.columns


# In[6]:


data01=data.drop(["zpid",'city',"zipcode","description","latitude","longitude", "latest_saledate",'latest_salemonth','latest_saleyear',
'latestPriceSource', 'numOfPhotos', 'homeImage' ,"streetAddress"], axis=1)


# In[7]:


data01.columns


# In[8]:


plt.figure(figsize=(10,10))
sns.histplot(data01['propertyTaxRate'])


# In[9]:


plt.figure(figsize=(10,10))
sns.histplot(data01['garageSpaces'])


# In[10]:


q = data01['garageSpaces'].quantile(0.95)
data02=data01[(data01['garageSpaces']<q)]


# In[11]:


plt.figure(figsize=(10,10))
sns.histplot(data02['garageSpaces'])


# In[12]:


data02['hasAssociation'].unique()


# In[13]:


data02['hasAssociation'].replace({True:1,False:0},inplace=True)


# In[14]:


data03=data02


# In[15]:


plt.figure(figsize=(10,10))
sns.histplot(data03['hasAssociation'])


# In[16]:


data03['hasCooling'].unique()


# In[17]:


data03['hasCooling'].replace({True:1,False:0},inplace=True)


# In[18]:


data04=data03


# In[19]:


plt.figure(figsize=(10,10))
sns.histplot(data04['hasCooling'])


# In[20]:


data04['hasGarage'].unique()


# In[21]:


data04['hasGarage'].replace({True:1,False:0},inplace=True)
data05=data04


# In[22]:


plt.figure(figsize=(10,10))
sns.histplot(data05['hasGarage'])


# In[23]:


data05['hasHeating'].unique()


# In[24]:


data05['hasHeating'].replace({True:1,False:0},inplace=True)
data06=data05


# In[25]:


plt.figure(figsize=(10,10))
sns.histplot(data06['hasHeating'])


# In[26]:


data06['hasSpa'].unique()


# In[27]:


data06['hasSpa'].replace({True:1,False:0},inplace=True)
data07=data06


# In[28]:


plt.figure(figsize=(10,10))
sns.histplot(data07['hasSpa'])


# In[29]:


data07['hasView'].unique()


# In[30]:


data07['hasView'].replace({True:1,False:0},inplace=True)
data08=data07


# In[31]:


plt.figure(figsize=(10,10))
sns.histplot(data08['hasView'])


# In[32]:


data08['homeType'].unique()


# In[33]:


plt.figure(figsize=(20,30))
sns.histplot(data08['homeType'])


# In[34]:


data08['homeType']=data08['homeType'].astype("category")


# In[35]:


data08['homeType']=data08['homeType'].cat.remove_categories(['Mobile / Manufactured','Other','MultiFamily','Residential','Apartment'])


# In[36]:


plt.figure(figsize=(30,20))
sns.histplot(data08['homeType'])


# In[37]:


data09=data08


# In[38]:


data09['parkingSpaces'].unique()


# In[39]:


plt.figure(figsize=(10,10))
sns.histplot(data09['parkingSpaces'])


# In[40]:


plt.figure(figsize=(15,10))
sns.histplot(data09['yearBuilt'])


# In[41]:


data09['latestPrice'].dtype


# In[95]:


plt.figure(figsize=(20,10))
sns.histplot(data09['latestPrice'],bins=30)


# In[43]:


q1= data09['latestPrice'].quantile(0.95)
data10=data09[(data09['latestPrice']<q1)]


# In[96]:


plt.figure(figsize=(10,10))
sns.histplot(data10['latestPrice'], bins=30)


# In[45]:


plt.figure(figsize=(10,10))
sns.histplot(data10['numPriceChanges'])


# In[46]:


q2= data10['numPriceChanges'].quantile(0.95)
data11=data10[(data10['numPriceChanges']<q2)]


# In[47]:


plt.figure(figsize=(10,10))
sns.histplot(data11['numPriceChanges'])


# In[48]:


plt.figure(figsize=(10,10))
sns.histplot(data11['numOfAccessibilityFeatures'])


# In[49]:


data12=data11.drop(['numOfAccessibilityFeatures'],axis=1)


# In[50]:


data12.columns


# In[51]:


plt.figure(figsize=(10,10))
sns.histplot(data12['numOfAppliances'])


# In[52]:


q3= data12['numOfAppliances'].quantile(0.95)
data13=data12[(data12['numOfAppliances']<q3)]


# In[53]:


plt.figure(figsize=(10,10))
sns.histplot(data13['numOfAppliances'])


# In[54]:


plt.figure(figsize=(10,10))
sns.histplot(data13['numOfParkingFeatures'])


# In[55]:


data13['numOfParkingFeatures'].unique()


# In[56]:


q4= data13['numOfParkingFeatures'].quantile(0.9999)
data14=data13[(data13['numOfParkingFeatures']<q4)]


# In[57]:


plt.figure(figsize=(10,10))
sns.histplot(data14['numOfParkingFeatures'])


# In[58]:


plt.figure(figsize=(10,10))
sns.histplot(data14['numOfPatioAndPorchFeatures'])


# In[59]:


q5= data14['numOfPatioAndPorchFeatures'].quantile(0.999)
data15=data14[(data14['numOfPatioAndPorchFeatures']<q5)]


# In[60]:


plt.figure(figsize=(10,10))
sns.histplot(data15['numOfPatioAndPorchFeatures'])


# In[61]:


plt.figure(figsize=(10,10))
sns.histplot(data15['numOfSecurityFeatures'])


# In[62]:


q6= data15['numOfSecurityFeatures'].quantile(0.995)
data16=data15[(data15['numOfSecurityFeatures']<q6)]


# In[63]:


plt.figure(figsize=(10,10))
sns.histplot(data16['numOfSecurityFeatures'])


# In[67]:


plt.figure(figsize=(10,10))
sns.histplot(data16['numOfWaterfrontFeatures'])


# In[70]:


data17=data16.drop(['numOfWaterfrontFeatures'],axis=1)


# In[72]:


plt.figure(figsize=(10,10))
sns.histplot(data17['numOfWindowFeatures'])


# In[75]:


q7= data17['numOfWindowFeatures'].quantile(0.998)
data18=data17[(data17['numOfWindowFeatures']<q7)]


# In[76]:


plt.figure(figsize=(10,10))
sns.histplot(data18['numOfWindowFeatures'])


# In[77]:


plt.figure(figsize=(10,10))
sns.histplot(data18['numOfCommunityFeatures'])


# In[82]:


data19=data18.drop(['numOfCommunityFeatures'],axis=1)


# In[100]:


plt.figure(figsize=(10,10))
sns.histplot(data19['lotSizeSqFt'],bins=10)


# In[110]:


q8= data19['lotSizeSqFt'].quantile(0.9)
data20=data19[(data19['lotSizeSqFt']<q8)]


# In[111]:


plt.figure(figsize=(10,10))
sns.histplot(data20['lotSizeSqFt'],bins=10)


# In[112]:


plt.figure(figsize=(10,10))
sns.histplot(data20['livingAreaSqFt'])


# In[113]:


q9= data20['livingAreaSqFt'].quantile(0.9)
data21=data20[(data20['livingAreaSqFt']<q9)]


# In[114]:


plt.figure(figsize=(10,10))
sns.histplot(data21['livingAreaSqFt'])


# In[115]:


plt.figure(figsize=(10,10))
sns.histplot(data21['numOfPrimarySchools'])


# In[116]:


plt.figure(figsize=(10,10))
sns.histplot(data21['numOfElementarySchools'])


# In[117]:


plt.figure(figsize=(10,10))
sns.histplot(data21['numOfMiddleSchools'])


# In[118]:


plt.figure(figsize=(10,10))
sns.histplot(data21['numOfHighSchools'])


# In[119]:


plt.figure(figsize=(10,10))
sns.histplot(data21['avgSchoolDistance'])


# In[120]:


q10= data21['avgSchoolDistance'].quantile(0.9)
data22=data21[(data21['avgSchoolDistance']<q10)]


# In[121]:


plt.figure(figsize=(10,10))
sns.histplot(data22['avgSchoolDistance'])


# In[122]:


plt.figure(figsize=(10,10))
sns.histplot(data22['avgSchoolRating'])


# In[123]:


plt.figure(figsize=(10,10))
sns.histplot(data22['avgSchoolSize'])


# In[124]:


plt.figure(figsize=(10,10))
sns.histplot(data22['MedianStudentsPerTeacher'])


# In[125]:


plt.figure(figsize=(10,10))
sns.histplot(data22['numOfBathrooms'])


# In[126]:


q11= data22['numOfBathrooms'].quantile(0.9)
data23=data22[(data22['numOfBathrooms']<q10)]


# In[127]:


plt.figure(figsize=(10,10))
sns.histplot(data23['numOfBathrooms'])


# In[128]:


plt.figure(figsize=(10,10))
sns.histplot(data23['numOfBedrooms'])


# In[157]:


q11= data23['numOfBedrooms'].quantile(0.995)
q12= data23['numOfBedrooms'].quantile(0.001)

data24=data23[(data23['numOfBedrooms']<q11) & (data23['numOfBedrooms']>q12)]


# In[158]:


plt.figure(figsize=(10,10))
sns.histplot(data24['numOfBedrooms'])


# In[163]:


plt.figure(figsize=(10,10))
sns.histplot(data24['numOfStories'])


# In[164]:


data24.describe()


# In[165]:


data24.columns


# In[172]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='propertyTaxRate',data=data24, multiple='stack')


# In[173]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='hasAssociation',data=data24, multiple='stack')


# In[174]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='hasGarage',data=data24, multiple='stack')


# In[182]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='avgSchoolRating',data=data24, multiple='stack')


# In[183]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='MedianStudentsPerTeacher',data=data24, multiple='stack')


# In[187]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='numOfBathrooms',data=data24, multiple='stack')


# In[188]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='numOfBedrooms',data=data24, multiple='stack')


# In[190]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='numOfAppliances',data=data24, multiple='stack')


# In[192]:


plt.figure(figsize=(25,60))
sns.kdeplot(x='latestPrice',hue='avgSchoolDistance',data=data24, multiple='stack')


# In[193]:


plt.figure(figsize=(25,100))
sns.kdeplot(x='latestPrice',hue='yearBuilt',data=data24, multiple='stack')


# In[195]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='garageSpaces',data=data24, multiple='stack')


# In[196]:


plt.figure(figsize=(25,25))
sns.kdeplot(x='latestPrice',hue='parkingSpaces',data=data24, multiple='stack')


# In[197]:


data_shuffle=data24.sample(frac=1)
data_shuffle=data_shuffle.reset_index(drop=True)


# In[198]:


x=data_shuffle.drop('latestPrice',axis=1)
y=data_shuffle['latestPrice']


# In[199]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder


# In[201]:


CT=make_column_transformer(
(MinMaxScaler(),['propertyTaxRate', 'garageSpaces', 'hasAssociation', 'hasCooling',
       'hasGarage', 'hasHeating', 'hasSpa', 'hasView',
       'parkingSpaces', 'yearBuilt', 'numPriceChanges',
       'numOfAppliances', 'numOfParkingFeatures', 'numOfPatioAndPorchFeatures',
       'numOfSecurityFeatures', 'numOfWindowFeatures', 'lotSizeSqFt',
       'livingAreaSqFt', 'numOfPrimarySchools', 'numOfElementarySchools',
       'numOfMiddleSchools', 'numOfHighSchools', 'avgSchoolDistance',
       'avgSchoolRating', 'avgSchoolSize', 'MedianStudentsPerTeacher',
       'numOfBathrooms', 'numOfBedrooms', 'numOfStories']),
 (OneHotEncoder(),['homeType'])
)


# In[202]:


CT.fit(x)
x_scaled=CT.transform(x)


# In[203]:


from sklearn.model_selection import train_test_split


# In[204]:


x_all_train,x_test,y_all_train,y_test=train_test_split(x_scaled,y,test_size=0.1,random_state=365)


# In[205]:


x_train,x_validation,y_train,y_validation=train_test_split(x_all_train,y_all_train,test_size=0.1,random_state=365)


# In[207]:


import tensorflow as tf


# In[208]:


train_inputs=tf.constant(x_train)
train_targets=tf.constant(y_train)

validation_inputs=tf.constant(x_validation)
validation_targets=tf.constant(y_validation)

test_inputs=tf.constant(x_test)
test_targets=tf.constant(y_test)


# In[243]:


early_stopping=tf.keras.callbacks.EarlyStopping(patience=7)


model=tf.keras.Sequential([
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1000, activation="relu"),
                     tf.keras.layers.Dense(1,activation="relu")
                    ])


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, 
                                                 patience=4,
                                                 verbose=1, 
                                                 min_lr=1e-8)

model.compile(optimizer="adam",loss="huber",metrics=["mae"])

model.fit(train_inputs,train_targets,batch_size=50,callbacks=[early_stopping, reduce_lr],
          validation_data=(validation_inputs, validation_targets),epochs=100, verbose=2)


# In[244]:


test_loss,test_accuracy=model.evaluate(test_inputs,test_targets)

