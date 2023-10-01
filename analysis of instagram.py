#!/usr/bin/env python
# coding: utf-8

# # Instagram reach analysis

# In[1]:


import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import svm
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\Requirements (1)\Instagram.csv", encoding='latin1')
df.head()


# In[3]:


df.isnull().sum()
df=df.dropna()
df.describe


# # Analyzing instagram reach

# In[4]:


plt.figure(figsize = (10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impression from Home")
sns.distplot(df['From Home'])
plt.show()


# In[5]:


plt.figure(figsize = (10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impression from Hashtags")
sns.distplot(df['From Hashtags'])
plt.show()


# In[6]:


plt.figure(figsize = (10,8))
plt.title("Distribution of Impression from Explore")
sns.distplot(df['From Explore'])
plt.show()


# In[7]:


home = df['From Home'].sum()
hashtags = df['From Hashtags'].sum()
explore = df['From Explore'].sum()
other = df['From Other'].sum()

labels = ['From Home', 'From Hashtags','From Explore','From other']
values = [home,hashtags,explore,other]
fig = px.pie(df,values = values,names = labels,
            title = 'Impression on Instagram Posts From various Sources', hole= 0.5)
fig.show()


# # Analyzing the content

# In[8]:


# text = '--'.join(i for i df.Caption)
text = '-'.join(str(i) for i in df['Caption'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords,background_color = 'white').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[9]:


text = '-'.join(str(i) for i in df['Hashtags'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords,background_color = 'white').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# # Analyzing relationship other

# In[10]:


text = '-'.join(str(i) for i in df['Hashtags'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords,background_color = 'white').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[11]:


figure = px.scatter(data_frame=df, x='Impressions', y='Likes',
                   size='Likes', trendline='ols',
                   title='Relation Between Likes and Impressions')
figure.show()


# In[12]:


figure = px.scatter(data_frame=df, x='Impressions', y='Comments',
                   size='Comments', trendline='ols',
                   title='Relation Between Comments and Impressions')
figure.show()


# In[13]:


figure = px.scatter(data_frame=df, x='Impressions', y='Shares',
                   size='Shares', trendline='ols',
                   title='Relation Between Shares and Impressions')
figure.show()


# In[14]:


correlation = df.corr()
print(correlation['Impressions'].sort_values(ascending= False))


# # Conversion rate

# In[16]:


con = (df['Follows'].sum()/df['Profile Visits'].sum())
print(con)


# In[17]:


fig = px.scatter(data_frame=df, x='Profile Visits', y='Follows',
                size='Follows', trendline='ols',
                title='Relationship Between Profile Visits and Followers Gained')
fig.show()


# # Predict the model

# In[18]:


x = np.array(df[['Likes','Saves','Comments','Shares','Profile Visits','Follows']])
y = np.array(df['Impressions'])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# In[19]:


from sklearn.linear_model import PassiveAggressiveRegressor


# # Check the accuracy r2_score

# In[21]:


model = PassiveAggressiveRegressor()
model.fit(x_train,y_train)

model.score(x_test,y_test)


# In[23]:


# Features [likes,shares,comments]
feat = np.array([[202.0,233.0,4.0,9.0,165.0,54.0]])
model.predict(feat)


# In[ ]:





# In[ ]:





# In[ ]:




