#!/usr/bin/env python
# coding: utf-8

# # K-NEAREST NEIGBOURS
# 

# In[1]:


# Given a training data ,model has been trained,classify based on features ,closet nearer neighbours K-NN


# In[2]:


# Euclidean distance,Manhatten distance


# In[3]:


# Lazy learner
# Pros and Cons


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/breast%20cancer.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[9]:


df = df.drop(['Unnamed: 32'],axis=1)


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.diagnosis.value_counts()


# In[14]:


df.diagnosis.value_counts()[0]


# In[15]:


df.diagnosis.value_counts()[1]


# In[19]:


sns.countplot(x='diagnosis',data=df,color='g')
plt.show()


# In[23]:


from sklearn.feature_selection import SelectKBest, f_classif


# In[24]:


df['diagnosis']= df['diagnosis'].replace({'M':1,'B':0})


# In[25]:


x= df.drop('diagnosis',axis=1)
y=df.diagnosis


# In[26]:


best_features = SelectKBest(score_func=f_classif, k=17)
fit =best_features.fit(x,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)

feature_scores= pd.concat([df_columns,df_scores],axis=1)
feature_scores.columns=['Feature_name','Score']
print(feature_scores.nlargest(17,'Score'))


# In[27]:


new_x = df[[ 'concave points_worst', 'perimeter_worst','concave points_mean','radius_worst','perimeter_mean','area_worst','radius_mean','area_mean','concavity_mean','concavity_worst','compactness_mean', 'radius_se', 'perimeter_se','area_se','texture_worst','smoothness_worst']]


# In[28]:


new_x


# In[30]:


scaler = StandardScaler()
x_scaler = scaler.fit_transform(new_x)


# In[32]:


from time import time
x_train,x_test,y_train,y_test = train_test_split(x_scaler,y,test_size = 0.25, random_state= 355)
knn = KNeighborsClassifier()

start = time()
knn.fit(x_train,y_train)
print('Knn training time: ', (time()--start))

start = time()
y_pred = knn.predict(x_test)
print('knn test time : ', (time()- start))


# In[33]:


cfm = confusion_matrix(y_test,y_pred)
cfm


# In[34]:


print(classification_report(y_test,y_pred,digits=2))


# In[36]:


from sklearn.model_selection import KFold,cross_val_score

k_f = KFold(n_splits=3,shuffle = True)
k_f


# In[38]:


for train, test in k_f.split([1,2,3,4,5,6,7,8,9,10]):
    print('train :',train,   'test :',test)


# In[39]:


cross_val_score(knn,x_scaler,y, cv=10)


# In[41]:


cross_val_score(KNeighborsClassifier(),x_scaler,y,cv=10).mean()


# In[42]:


# Hyperparameter Tuning


# In[43]:


from sklearn.model_selection import GridSearchCV


# In[45]:


param_grid = {'algorithm'  : ['kd_tree', 'brute'],
             'leaf_size' : [6,7,8,10,11,14],
             'n_neighbors' : [3,5,7,9,11,13]}


# In[46]:


gridsearch = GridSearchCV(estimator=knn, param_grid=param_grid)


# In[47]:


gridsearch.fit(x_train,y_train)


# In[48]:


gridsearch.best_params_


# In[50]:


knn = KNeighborsClassifier(algorithm = 'kd_tree', leaf_size = 6, n_neighbors = 3)


# In[51]:


knn.fit(x_train,y_train)


# In[52]:


y_pred = knn.predict(x_test)


# In[53]:


cfm = confusion_matrix(y_test,y_pred)
cfm


# In[54]:


print(classification_report(y_test,y_pred, digits=3))


# In[80]:


import pandas as pd


# In[81]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru'],
                  'Gender':['Male','Female','Female','Female','Male','Male'],
                  'Exp':[1,3,5,6,9,None]})
df


# In[82]:


from sklearn.preprocessing import LabelEncoder


# In[83]:


lab_enc = LabelEncoder()


# In[84]:


df2 = lab_enc.fit_transform(df['City'])
pd.Series(df2)


# In[85]:


df['City']= df2
df


# In[86]:


df


# In[87]:


df2


# In[88]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru'],
                  'Gender':['Male','Female','Female','Female','Male','Male'],
                  'Exp':[1,3,5,6,9,None]})
df


# In[99]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


# In[100]:


import pandas as pd


# In[101]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru'],
                  'Gender':['Male','Female','Female','Female','Male','Male'],
                  'Exp':[1,3,5,6,9,None]})
df


# In[105]:


ohe = OneHotEncoder()
si = SimpleImputer


# In[106]:


ct = make_column_transformer(
    (ohe,['City','Gender']),
    (si,['Exp']),
     remainder='passthrough')    # 'passthrough' to keep all other columns


# In[107]:


encoded = pd.DataFrame(ct.fit_transform(df))
encoded


# In[108]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


# In[109]:


ohe = OneHotEncoder()
si = SimpleImputer()


# In[110]:


import pandas as pd


# In[111]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru'],
                  'Gender':['Male','Female','Female','Female','Male','Male'],
                  'Exp':[1,3,5,6,9,None]})
df


# In[112]:


ct = make_column_transformer(
    (ohe,['City','Gender']),
    (si,['Exp']),
     remainder='passthrough') 


# In[113]:


encoded= pd.DataFrame(ct.fit_transform(df))
encoded


# In[114]:


# remane the columns as per our choice


# In[116]:


encoded = pd.DataFrame(ct.fit_transform(df),columns=['City_Bengaluru','City_Delhi','City_Hyd','Gender_Female','Gender_male','Exp','Salary'])


# In[117]:


encoded


# In[118]:


df  # original Data set


# #  get_dummies

# one hot Encoding and get_dummies almost equal. Major differences is if you want to reduce (drop_first= true) the column size of the dataset you can use get_dummies.

# In[121]:


df1 = pd.get_dummies(df[['City','Gender']],drop_first=False)
df1


# # Ordinal Encoder

# In[122]:


from sklearn.preprocessing import OrdinalEncoder


# In[123]:


import pandas as pd

Employee = pd.DataFrame({'Position':['SE','Manager','Team Lead','SSE'],
                        'Project':['A','B','C','D'],
                        'Salary':[25000,85000,71000,48000]})
Employee


# In[124]:


ord_enc = OrdinalEncoder(categories=[['SE','SSE','Team Lead','Manager'],['A','B','C','D']])
Encoded_df = ord_enc.fit_transform(Employee[['Position','Project']])


# In[125]:


Encoded_df


# # Binary Encoder

# In[126]:


import pandas as pd


# In[127]:


df = pd.DataFrame({'Cat_data':['A','B','C','D','E','F','G','H','I','A','A','D']})
df


# In[133]:


get_ipython().system('pip install category_encoders')


# In[134]:


from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder


# In[135]:


bi_enc = BinaryEncoder()


# In[136]:


df_bi = bi_enc.fit_transform(df)
df_bi


# # KNN imputer

# # iterative imputer

# In[137]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000,None],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru','Hyderabad'],
                  'Gender':['Male','Female','Female','Female','Male','Male','Male'],
                  'Exp':[1,3,5,6,9,4,11]})


# In[138]:


df


# In[140]:


from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer


# In[141]:


df = pd.DataFrame({'salary':[25000,48000,71000,85000,90000,55000,None],
                  'City':['Bengaluru','Delhi','Hyderabad','Bengaluru','Hyderabad','Bengaluru','Hyderabad'],
                  'Gender':['Male','Female','Female','Female','Male','Male','Male'],
                  'Exp':[1,3,5,6,9,4,11]})
df


# In[142]:


iter_impute = IterativeImputer()
ite_imp = pd.DataFrame(iter_impute.fit_transform(df[['salary','Exp']]),columns=['salary','Exp'])
ite_imp


# # data scientist life cycle

# In[144]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# In[145]:


data = pd.read_csv( 'https://raw.githubusercontent.com/training-ml/Files/main/wine.csv')
data.head()


# In[146]:


data.describe()


# In[147]:


data.isna().sum()


# In[148]:


from sklearn.preprocessing import OrdinalEncoder


# In[149]:


ord_encoder = OrdinalEncoder(categories=[['Low','Medium','High']])

df1 = ord_encoder.fit_transform(data[['Alcohol_content']])
df1


# In[150]:


data['Alcohol_content']= df1


# In[151]:


data.head()


# In[152]:


data.describe


# In[154]:


df = pd.DataFrame({'Cat_data':['A','B','C','D','E','F','G','H','I','A','A','D']})


# In[155]:


df


# In[156]:


d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df


# In[157]:


df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])


# In[158]:


df2


# In[161]:


df3= pd.DataFrame(np.array([[1,2,3],[4,5,6],[7,8,9]]),
columns=['chandu','Eshan','sandhya'])
df3


# In[164]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[165]:


df= pd.read_csv('NetflixBestOf_reddit.csv')
df


# In[166]:


df.describe()


# In[167]:


df.info()


# In[168]:


df['score']


# In[170]:


df['title'][12]


# In[174]:


df['score'][13]


# In[184]:


df = pd.DataFrame({'Name':['chandu','Eshan','Sandhya'],
                  'Age':[39,7,35],
                  'sex':['Male','Male','Female'],
                  'Salary':[180000,150000,140000]})
df


# In[185]:


df['Age'].max()


# In[186]:


df.describe()


# In[187]:


df[df['Age']>35]


# In[188]:


df.plot()


# In[ ]:




