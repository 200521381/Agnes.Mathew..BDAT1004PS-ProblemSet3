#!/usr/bin/env python
# coding: utf-8

# Question 1

# In[7]:


#Step 1. Import the necessary libraries
import urllib
import requests
import urllib.request
import pandas as pnds
import numpy as nmp


# In[9]:


#Step 2. Import the dataset
#Step 3. Assign it to a variable called users
users=pnds.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')
users


# In[10]:


response = requests.get('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user')
print(response.status_code)


# In[11]:


#Step 4. Discover what is the mean age per occupation
users.groupby('occupation').age.mean()


# In[12]:


#Step 5. Discover the Male ratio per occupation and sort it from the most to the least 
gender_total = users['gender'].value_counts()
gender_occupation = users.groupby('gender')['occupation'].value_counts()
sex_ratio = gender_occupation['M']/gender_occupation['F']
print(sex_ratio.sort_values(ascending=False))


# In[16]:


#Step 6. For each occupation, calculate the minimum and maximum ages
users.groupby('occupation').age.agg(['min', 'max'])


# In[17]:


#Step 7. For each combination of occupation and sex, calculate the mean age
users.groupby(['occupation', 'gender']).age.mean()


# In[18]:


#Step 8. For each occupation present the percentage of women and men

occup_by_gender = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
occup_count = users.groupby(['occupation']).count()
occup_gender = occup_by_gender.div(occup_count, level = "occupation")
occup_gender.loc[:, 'gender']


# Question 2

# In[20]:


#Step 1. Import the necessary libraries -> Is already done, not required again
#Step 2. Import the dataset and Step 3. Assign it to a variable called euro12 
euro12=pnds.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv")
euro12


# In[21]:


#Step 4. Select only the Goal column
euro12.Goals


# In[22]:


#Step 5. How many team participated in the Euro2012?
euro12.Team.value_counts().count()


# In[23]:


#Step 6. What is the number of columns in the dataset?
euro12.shape[1]


# In[27]:


#Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
discipline=euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline


# In[30]:


#Step 8. Sort the teams by Red Cards, then to Yellow Cards
discipline.sort_values(by=['Red Cards','Yellow Cards'],ascending=False)
discipline


# In[31]:


#Step 9. Calculate the mean Yellow Cards given per Team
round(discipline['Yellow Cards'].mean())


# In[32]:


#Step 10. Filter teams that scored more than 6 goalsStep 11. Select the teams that start with G
euro12[euro12.Team.str.startswith('G')]


# In[33]:


#Step 12. Select the first 7 columns
euro12.iloc[:, 0:7]


# In[34]:


#Step 13. Select all columns except the last 3
euro12.iloc[:, :-3]


# In[35]:


#Step 14. Present only the Shooting Accuracy from England, Italy and Russia
euro12.loc[euro12.Team.isin(['England','Italy','Russia']),['Team','Shooting Accuracy']]


# Question 3

# In[36]:


#Step 1. Import the necessary libraries
import random
import numpy as nmp
import pandas as pnds


# In[37]:


#Step 2. Create 3 differents Series, each of length 100
first = pnds.Series(random.randint(1,4) for _ in range(100))
second = pnds.Series(random.randint(1,3) for _ in range(100))
third = pnds.Series(random.randint(1000,3000) for _ in range(100))


# In[38]:


#Step 3. Create a DataFrame by joinning the Series by column
data = pnds.concat([first, second, third], axis = 1)
data


# In[39]:


#Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
data.columns = ["bedrs", "bathrs", "price_sqr_meter"]
data


# In[41]:


#Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'
data2 = pnds.concat([first, second, third])
data2


# In[ ]:


#Step 6. Ops it seems it is going only until index 99. Is it true?
# True


# In[42]:


#Step 7. Reindex the DataFrame so it goes from 0 to 299
data2.reset_index(drop=True)


# Question 4

# In[43]:


# Step 1. Import the necessary libraries
import pandas as pnds


# In[45]:


# Step 2. Import the dataset from the attached file wind.txt
data=pnds.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\wind.txt")
data


# In[47]:


# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index
data = pnds.read_table('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data', sep='\s+', parse_dates=[[0,1,2]])
data


# In[48]:


# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it
import datetime
def date_fix(d):
    year = d.year - 100 if d.year > 1989 else d.year
    return datetime.date(year, d.month ,d.day)
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(date_fix)
data


# In[50]:


#Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64
data["Yr_Mo_Dy"] = pnds.to_datetime(data["Yr_Mo_Dy"])
data = data.set_index('Yr_Mo_Dy')
data


# In[51]:


# Step 6. Compute how many values are missing for each location over the entire record
data.isnull().sum()


# In[52]:


# Step 7. Compute how many non-missing values there are in total
data.notnull().sum()


# In[53]:


# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times
x = data.mean()
x.mean()


# In[56]:


# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the
#days
def stats(x):
    x = pnds.Series(x)
    Min = x.min()
    Max = x.max()
    Mean = x.mean()
    Stnd = x.std()
    res = [Min,Max,Mean,Stnd]
    indx = ["Min","Max","Mean","Stnd"]
    res = pnds.Series(res,index=indx)
    return res
loc_stats = data.apply(stats)
loc_stats


# In[57]:


data.describe(percentiles=[])


# In[58]:


# Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each
#day
day_stats = pnds.DataFrame()
day_stats['min'] = data.min(axis = 1) 
day_stats['max'] = data.max(axis = 1) 
day_stats['mean'] = data.mean(axis = 1) 
day_stats['std'] = data.std(axis = 1)
day_stats.head()


# In[59]:


# Step 11. Find the average windspeed in January for each location
data.loc[data.index.month == 1].mean()


# In[60]:


# Step 12. Downsample the record to a yearly frequency for each location
data.groupby(data.index.to_period('A')).mean()


# In[61]:


# Step 13. Downsample the record to a monthly frequency for each location
data.groupby(data.index.to_period('M')).mean()


# In[62]:


# Step 14. Downsample the record to a weekly frequency for each location
data.groupby(data.index.to_period('W')).mean()


# In[63]:


# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week
first_year = data[data.index.year == 1961]
data1 = data.resample('W').mean().apply(lambda x: x.describe())
print (data1)


# Question 5

# In[64]:


# Step 1. Import the necessary libraries
import pandas as pnds


# In[66]:


# Step 2. Import the dataset and Step 3. Assign it to a variable called chipo
chipo=pnds.read_table("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv")
chipo


# In[67]:


# Step 4. See the first 10 entries
chipo.head(10)


# In[68]:


# Step 5. What is the number of observations in the dataset?
chipo.shape[0]


# In[70]:


# Step 6. What is the number of columns in the dataset?
len(chipo.columns)


# In[71]:


# Step 7. Print the name of all the columns
chipo.columns


# In[72]:


# Step 8. How is the dataset indexed?
chipo.index


# In[73]:


# Step 9. Which was the most-ordered item?
most_ordr = chipo.groupby(['item_name']).agg({'quantity':'sum'})
most_ordr.sort_values('quantity',ascending=False)[:5]


# In[74]:


# Step 10. For the most-ordered item, how many items were ordered?
most_ordr = chipo.groupby(['item_name']).agg({'quantity':'sum'})
most_ordr.sort_values('quantity',ascending=False)


# In[75]:


# Step 11. What was the most ordered item in the choice_description column?
most_ordr = chipo.groupby(['choice_description']).agg({'quantity':'sum'})
most_ordr.sort_values('quantity',ascending=False)[:5]


# In[76]:


# Step 12. How many items were orderd in total?
chipo.quantity.sum()


# In[77]:


# Step 13
chipo.item_price.str.slice(1).astype(float).head()


# In[93]:


chipo.info()


# In[ ]:


lmb = lambda x : float(x[1:])
chipo.item_price.apply(lmb)[:5]


# In[ ]:


chipo['item_price']=chipo.item_price.apply(lmb)


# In[94]:


# Step 14. How much was the revenue for the period in the dataset?
chipo['item_price'].sum()


# In[95]:


# Step 15. How many orders were made in the period?
chipo.shape


# In[96]:


# Step 16. What is the average revenue amount per order?
chipo['item_price'].mean()


# In[97]:


# Step 17. How many different items are sold?
chipo.item_name.nunique()


# Question 6

# In[99]:


# Importing libraries and dataset
import pandas as pd
data=pd.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\us-marriages-divorces-1867-2014.csv")
data


# In[104]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
data=pd.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\us-marriages-divorces-1867-2014.csv")

marriagesPerCapita = data['Marriages_per_1000']
divorcesPerCapita = data['Divorces_per_1000']
columns = marriagesPerCapita, divorcesPerCapita
years = data['Year']
indx = ["marriages per capita", "divorces per capita"]
x_data = range(1867, 1867 + data.shape[0])
fig, ax = plt.subplots()
for i in range(len(columns)):
    ax.plot(x_data, columns[i], label=ind[i])

ax.set_xlabel('Year')
ax.set_ylabel('Marriage/Divorces per 1000 people')
ax.legend()


# Question 7

# In[107]:


marriagesPerCapita = data['Marriages_per_1000']
divorcesPerCapita = data['Divorces_per_1000']

year = [1900, 1950, 2000]
indx = ["marriages per capita", "divorces per capita"]

specific_data = data.loc[data['Year'] == 1900]
specific_data = specific_data.append(data.loc[data['Year'] == 1950])
specific_data = specific_data.append(data.loc[data['Year'] == 2000])
columns = specific_data['Marriages_per_1000'], specific_data['Divorces_per_1000']
data = data[['Year','Marriages_per_1000','Divorces_per_1000']]
data

import matplotlib.pyplot as plt
x = ["1900","1950","2000"]
Marriages = [9.3,11.0,8.2]
Divorces = [0.7,2.5,3.3]

plt.bar(x,Marriages,0.2,label="Marriages")
plt.bar(x,Divorces,0.2,label="Divorces")
       
plt.xlabel("year")
plt.ylabel("Marriages/Divorces people")
plt.legend()
plt.show()


# Question 8

# In[108]:


# Importing libraries and dataset
import matplotlib.pyplot as plt

Hollywood_actors = pd.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\actor_kill_counts.csv")
Hollywood_actors


# In[109]:


m =Hollywood_actors['Actor'].values
n =Hollywood_actors['Count'].values
plt.xticks(rotation='vertical')
plt.xlabel('Number of kills')
plt.ylabel('Actors')
plt.barh(m,n)
plt.show()


# Question 9

# In[110]:


# Importing libraries and dataset
import pandas as pnds

data=pnds.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\roman-emperor-reigns.csv")
data


# In[119]:


roman_emperors = pnds.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\roman-emperor-reigns.csv")
assassinated_emperors = roman_emperors[
    roman_emperors['Cause_of_Death'].apply(lambda x: 'assassinated' in x.lower())]

number_assassinated = len(assassinated_emperors)
other_deaths = len(roman_emperors) - number_assassinated

plt.pie([other_deaths, 100-other_deaths], labels=['Assissanated', 'Not Assissanated'], autopct='%1.2f%%')
roman_emperors = pnds.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\roman-emperor-reigns.csv")
assassinated_emperors = roman_emperors.loc[data['Cause_of_Death'] == 'Assassinated']
roman_emperor_reign_lengths = roman_emperors['Length_of_Reign'].values

assassinated_emperors = assassinated_emperors.append(data.loc[data['Cause_of_Death'] == 'Possibly assassinated'])
plt.pie(assassinated_emperors['Length_of_Reign'], labels=assassinated_emperors['Emperor'], autopct='%1.2f%%')


# Question 10

# In[122]:


# import libraries and dataset
import pandas as pd
data = pd.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\arcade-revenue-vs-cs-doctorates.csv")
data


# In[126]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
data1 = pd.read_csv(r"F:\BDAT\Data Programming\Assignment\Problem Set 3\arcade-revenue-vs-cs-doctorates.csv")
arcade_revenue = data1['Total Arcade Revenue (billions)'].values
cs_doctorates_awarded = data1['Computer Science Doctorates Awarded (US)'].values
fig, ax = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, len(data1['Year'])))
for i in range(len(data1['Year'])):
    ax.scatter(arcade_revenue[i], cs_doctorates_awarded[i],color=colors[i])
ax.set_xlabel('Total Arcade Revenue in (billions)')
ax.set_ylabel('Computer Science Awarded in (US)')


# In[ ]:




