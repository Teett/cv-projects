#%%
import pysubgroup as ps
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%%
# Define a dictionary that maps each column to its data type

# Data credits: https://archive-beta.ics.uci.edu/dataset/143/statlog+australian+credit+approval
dtypes = {'A1': 'category', 'A2': 'float', 'A3': 'float', 'A4': 'category',
          'A5': 'category', 'A6': 'category', 'A7': 'float', 'A8': 'category',
          'A9': 'category', 'A10': 'float', 'A11': 'category', 'A12': 'category',
          'A13': 'float', 'A14': 'float', 'A15': 'int'}
# Read the .dat file and specify the separator as a space
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat', 
                   sep=' ', names=list(dtypes.keys()), dtype=dtypes)

data['A15'] = data['A15'].replace({0: 'Declined', 1: 'Approved'})
#%%
## What does our df look like?
data.head()
## What are the dtypes?
data.dtypes
## Check the nulls!
data.isna().sum()

# %%
# Explore the dataframe
# Numeric features | Make a pairplot with seaborn!


# %% Focus on A3 vs A7
plt.figure(figsize=(8, 10))
sns.scatterplot(x='A3', y='A7', style='A15', data=data, edgecolor='black')
plt.legend(loc='upper left', bbox_to_anchor=(1.10, 1), borderaxespad=0)
plt.xlabel("A3 (Continuous)", fontsize=12)
plt.ylabel("A7 (Continuous)", fontsize=12)
plt.show()
# %% Use the subgroup discovery package 



#%%
# Investigate the initial interesting subgroups!

data['subgroup'] = 0
data.loc[data['A8'] == '1', 'subgroup'] = 1
data.loc[(data['A12'] == '2') & (data['A9'] == '1'), 'subgroup'] = 2

sns.pairplot(data.query('subgroup == 1'), hue='A15', vars=['A2', 'A3', 'A7'])
plt.show()

#%%
colors = {0: 'white', 1: 'red', 2: 'blue'}
plt.figure(figsize=(8, 10))
sns.scatterplot(x='A3', y='A7', style='A15', hue='subgroup', 
                palette=colors, data=data, edgecolor='black')
plt.legend(loc='upper left', bbox_to_anchor=(1.10, 1), borderaxespad=0)
plt.xlabel("A3 (Continuous)", fontsize=12)
plt.ylabel("A7 (Continuous)", fontsize=12)
plt.show()

#%%
# Are the means for features A2, A3 and A7 different on subgroups 1 and 2 vs no subgroup? 
data.groupby('subgroup')['A2', 'A3', 'A7'] \
     .agg(['mean', 'count'])

