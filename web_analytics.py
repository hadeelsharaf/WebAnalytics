#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import Pandas
import pandas as pd


# In[3]:


#load dataframe
wa_df =  pd.read_csv('./online_shoppers_intention.csv')


# In[6]:


wa_df.count()


# In[8]:


wa_df.shape


# In[9]:


wa_df.describe()


# In[ ]:





# In[10]:


wa_df.head()


# In[11]:


wa_df['VisitorType'].value_counts()


# In[12]:


wa_df['Revenue'].value_counts()


# In[14]:


wa_df.isnull().sum()


# In[4]:


# import Plotting tools to explore the data.
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(8,5))
total = float(len(wa_df))
ax = sns.countplot(x="Revenue", data=wa_df)
for p in ax.patches:
    percent = '{:.1f}'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percent,(x,y),ha='center')
plt.show()


# In[10]:


ax = sns.countplot(x="VisitorType", data=wa_df)
for p in ax.patches:
    percent = '{:.1f}'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percent,(x,y),ha='center')
plt.show()


# In[5]:


x,y = 'VisitorType', 'Weekend'
df1 = wa_df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
  txt = str(p.get_height().round(2)) + '%'
  txt_x = p.get_x()
  txt_y = p.get_height()
  g.ax.text(txt_x,txt_y,txt)


# In[21]:


x,y = 'TrafficType', 'Revenue'
#df1 = wa_df.query('Revenue==True')
df1 = wa_df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
  txt = str(p.get_height().round(2)) + '%'
  txt_x = p.get_x()
  txt_y = p.get_height()
  g.ax.text(txt_x,txt_y,txt)


# In[27]:


df1 = wa_df.query('Revenue==True')
total = float(len(df1))
ax = sns.countplot(x="TrafficType", data=df1)
for p in ax.patches:
    percent = '{:.1f}'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percent,(x,y),ha='center')
plt.show()


# In[28]:


plt.hist(wa_df['TrafficType'])
plt.title('Distribution of diff Traffic',fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[29]:


plt.hist(wa_df['Region'])
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[31]:


df1 = wa_df.query('Revenue==True')
plt.hist(df1['Region'])
plt.title('Dist. of Customers revenue',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[36]:


plt.hist(df1['OperatingSystems'])
plt.title('OS Distribution of Customers 2',fontsize = 30)
plt.xlabel('OperatingSystems', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[42]:


plt.hist(df1['Month'])
plt.title('Dist. of buyers per Month',fontsize = 28)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[44]:


sns.stripplot(data = wa_df, x='Revenue', y='PageValues')


# In[52]:


#sns.stripplot(data = wa_df, x='Revenue', y='BounceRates')
#sns.color_palette("YlOrBr", as_cmap=True)
sns.relplot(data = df1, x='ProductRelated_Duration', y='BounceRates',hue="ProductRelated")


# In[32]:


sns.lmplot(x = 'Administrative', y = 'Informational', data = wa_df, x_jitter = 0.05)


# In[51]:


plt.figure(figsize=(8,5))
sns.boxplot(x = wa_df['VisitorType'], y = wa_df['BounceRates'], hue = wa_df['Revenue'], palette = 'Blues')
plt.title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 25)


# In[55]:


sns.boxplot(x = wa_df['Month'], y = wa_df['BounceRates'],
            hue = wa_df['Revenue'], palette = 'Oranges')
plt.title('Month vs BounceRates w.r.t. Rev.', fontsize = 26)


# In[7]:


print(wa_df.iloc[:,[5,17]])
x = wa_df.iloc[:,[1,6]].values


# In[78]:


# inport Kmeans to apply clustering and elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    km = KMeans(n_clusters = i,
    init = 'k-means++',
    max_iter = 300,
    n_init = 10,
    random_state = 0,
    algorithm = 'full',
    tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 10), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[81]:


km = KMeans(n_clusters = 3, init = 'k-means++', 
            max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], 
            s = 100, c = 'red', label = 'Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], 
            s = 100, c = 'yellow', label = 'General Customers')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], 
            s = 100, c = 'green', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:, 1], 
            s = 50, c = 'blue' , label = 'centeroid')
plt.title('Administrative Duration vs Duration', fontsize = 20)
plt.grid()
plt.xlabel('Administrative Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()


# In[19]:


from sklearn.cluster import KMeans
#product duration and revenue Elbow
x = wa_df.iloc[:,[5,6]].values
wcss = []
for i in range(1, 10):
    km = KMeans(n_clusters = i,
    init = 'k-means++',
    max_iter = 300,
    n_init = 10,
    random_state = 0,
    algorithm = 'full',
    tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 10), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[25]:


# plotting clusters
import numpy as np

km = KMeans(n_clusters = 3, init = 'k-means++', 
            max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Getting unique labels
u_labels = np.unique(y_means)
for i in u_labels:
    plt.scatter(x[y_means == i , 0] , x[y_means == i , 1] , label = i)

plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:, 1], 
            s = 50, c = 'blue' , label = 'centeroid')
plt.title('Product Duration vs bounce rate', fontsize = 20)
plt.grid()
plt.xlabel('Product Duration')
plt.ylabel('bounce rate')
plt.legend()
plt.show()


# In[53]:


# one hot encoding
data1 = pd.get_dummies(wa_df)
print(data1.columns)


# In[54]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
wa_df['Revenue'] = le.fit_transform(wa_df['Revenue'])
print(wa_df['Revenue'].value_counts())


# In[55]:


# getting dependent and independent variables
x=data1
# removing the target column revenue from 
x = x.drop(['Revenue'], axis = 1)
y = data1['Revenue']
# checking the shapes
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[57]:


# splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                     test_size = 0.3, random_state = 0)
# checking the shape
print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)


# In[58]:


# MODELLING
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[59]:


# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))


# In[60]:


# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (6, 6)
sns.heatmap(cm ,annot = True)


# In[61]:


# classification report
cr = classification_report(y_test, y_pred)
print(cr)


# In[65]:


# ROC curve
from sklearn.metrics import plot_roc_curve
rf_disp = plot_roc_curve(model, x_test, y_test)
plt.show()


# In[69]:


df=pd.DataFrame(y_pred,columns=["Revenue"])
df


# In[70]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)


# In[71]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", cm)


# In[72]:


sns.heatmap(cm ,annot = True)


# In[74]:


cm = confusion_matrix(y, model.predict(x))
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# In[75]:


# classification report
cr1 = classification_report(y_test, y_pred1)
print(cr1)


# In[76]:


# ROC curve
from sklearn.metrics import plot_roc_curve
lr_disp = plot_roc_curve(model1, x_test, y_test)
plt.show()


# In[77]:


#predected results
dfl = pd.DataFrame(y_pred1, columns=["Revenue"])
dfl


# In[78]:


# combined ROC curve for comparing 
ax = plt.gca()
rf_disp = plot_roc_curve(model, x_test, y_test, ax=ax, alpha=0.8)
lr_disp.plot(ax=ax, alpha=0.8)
plt.show()


# In[ ]:




