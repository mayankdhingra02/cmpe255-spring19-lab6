
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from numpy.polynomial.polynomial import polyfit


# In[4]:


n1=np.array([17.5,22,29.5,44.5,64.5,80])
n2=np.array([38,36,24,20,18,28])


# In[5]:


n=np.column_stack((n1,n2))


# In[6]:


l=len(n1)
p=[0]*(l+1)


# In[7]:


lower_val=[]
upper_val=[]
for i in range(l):
    print("i=",i)
    if(i==0):
        r=int((n1[0]+n1[1])/2)
        lower=n1[0]-(r-n1[0])
    else:
        lower=r+1
        r=n1[i]+(n1[i]-lower)
    lower_val.append(lower)
    upper_val.append(r)
    print("lower =",lower," right= ",r)


# In[8]:


plt.scatter(n1,n2,s=20)
plt.title('Scatter plot')
plt.xlabel('Independent variable - age')
plt.ylabel('Dependent variablee - Deaths')


# In[9]:


df = pd.DataFrame(n, columns = ['Age', 'Deaths']) 


# In[10]:


age_sq=[]
death_sq=[]
xCrossY=[]
for i in range(l):
    age_sq.append(df['Age'][i]*df['Age'][i])
    death_sq.append(df['Deaths'][i]*df['Deaths'][i])
    xCrossY.append(df['Deaths'][i]*df['Age'][i])


# In[11]:


df['lower']=lower_val
df['upper']=upper_val

df['Age_sq']=age_sq
df['Death_sq']=death_sq
df['xCrossY']=xCrossY


# In[12]:


df=df[['Age','lower','upper','Deaths','Age_sq','Death_sq','xCrossY']]
df


# In[13]:


##Lets suppose y=b1*x + b0 is our linear line equation


# In[14]:


s_age=sum(df['Age'])
s_deaths=sum(df['Deaths'])
s_age_sq=sum(df['Age_sq'])
s_xy=sum(df['xCrossY'])


# In[15]:


b1=( s_xy - (s_age*s_deaths)/l ) / (s_age_sq  - (s_age*s_age)/l)
b0= ( s_deaths - (b1*s_age) )/l


# In[16]:


print("b1=",b1)
print("b0=",b0)


# In[17]:


#The final linear regression line turns out to be y=35.58 - 0.19*x


# In[18]:


#Predicting number of deaths for age 40 and 60


# In[19]:


#for age 40
y=b1*40 + b0
print(y)


# In[20]:


#for age 60
y=b1*60 + b0
print(y)


# In[21]:


plt.scatter(n1,n2,s=20)
plt.title('Scatter plot')
plt.xlabel('Independent variable - age')
plt.ylabel('Dependent variablee - Deaths')
x = n1
y = b1 * x + b0
b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()


# In[22]:


# There is a linear relationship between age and number of deaths upto age 74.
# The last age group is above the fitted line showing an increase from the previous values which makes it incosistent


# In[31]:


print(stats.pearsonr(n1,n2))


# In[ ]:


# For four df and alpha = 0.05, the LinRegTTest gives p-value = 0.2288 so we do
# not reject the null hypothesis; there is not a significant linear relationship
# between deaths and age.
# Using the table of critical values for the correlation coefficient, with four df,
# the critical value is 0.811. The correlation coefficient r = -0.57874 is not less
# than -0.811, so we do not reject the null hypothesis.


# In[32]:


print(y)


# In[33]:


print(np.corrcoef(x, y))


# In[34]:


df.plot()

