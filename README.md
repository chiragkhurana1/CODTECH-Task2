# CODTECH-Task2
EDA
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
os.chdir("//Users//chiragkhurana//Desktop")

inp0=pd.read_csv("appp.csv")
inp0.head()

prev=pd.read_csv('previous_application.csv')
prev.head()

# Handling and cleaning data

### checking for null values:

inp0.isnull().sum().head(20)

#Removing null values:
inp0=inp0[~inp0.AMT_ANNUITY.isnull()]
          
inp0.isnull().sum().head(20)

inp0=inp0[~inp0.AMT_GOODS_PRICE.isnull()]
inp0.isnull().sum().head(20)

inp0=inp0[~inp0.NAME_TYPE_SUITE.isnull()]
inp0.isnull().sum().head(20)

#### Checking for common values in NAME_CONTRACT_TYPE:



### What type of laon have target applied for:
target_counts = inp0['TARGET'].value_counts()
contract_counts = inp0['NAME_CONTRACT_TYPE'].value_counts()
combination_counts = inp0.groupby(['TARGET', 'NAME_CONTRACT_TYPE']).size()
df_loans=pd.DataFrame(combination_counts)



inp0['NAME_CONTRACT_TYPE'].value_counts()

prev['NAME_CONTRACT_TYPE'].value_counts()

## checking for outliers

### checking for outliers in income total..
plt.boxplot(inp0.AMT_INCOME_TOTAL)
plt.show()

inp0.AMT_INCOME_TOTAL.mean()

### income more than 150000
inp0[inp0.AMT_INCOME_TOTAL>150000].AMT_INCOME_TOTAL.plot.box()
plt.show()

### income more than 50000
inp0[inp0.AMT_INCOME_TOTAL<50000].AMT_INCOME_TOTAL.plot.box()
plt.show()


target_counts = inp0['TARGET'].value_counts(normalize=True) * 100 

### bar chart for distribution of the 'Target' variable
plt.figure(figsize=(8, 6))
target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Target')
plt.ylabel('Percentage')
plt.title('Distribution of Target Variable')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


### Mean Target Value by Contract Type
target_by_contract_type = inp0.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean()

plt.figure(figsize=(8, 6))
target_by_contract_type.plot(kind='bar', color='skyblue')
plt.xlabel('Contract Type')
plt.ylabel('Mean Target Value')
plt.title('Mean Target Value by Contract Type')
plt.xticks(rotation=45)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

sns.distplot(inp0.AMT_ANNUITY, bins=20)

plt.show()

# Analysing the previous applications 

### contract type analyse in previous applictions 
contract_type_counts =prev['NAME_CONTRACT_TYPE'].value_counts()
plt.figure(figsize=(8, 6))
contract_type_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Contract Type')
plt.ylabel('Frequency')
plt.title('Distribution of Contract Types')
plt.xticks(rotation=45)  
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.show()

plt.scatter(prev['NAME_CONTRACT_TYPE'], prev['AMT_ANNUITY'], color='skyblue', alpha=0.5)
plt.xlabel('NAME_CONTRACT_TYPE')
plt.ylabel('AMT_ANNUITY')
plt.title('Correlation of Contract type and amount annuity of previous applications')


# comparing it with new applictaions...

plt.scatter(inp0['NAME_CONTRACT_TYPE'], inp0['AMT_ANNUITY'], color='red', alpha=0.5)
plt.xlabel('NAME_CONTRACT_TYPE')
plt.ylabel('AMT_ANNUITY')
plt.title('Correlation of Contract type and amount annuity of recent applications')


####  top correlation by segmenting the data frame w.r.t to the target variable 

columns_to_keep = ['TARGET','AMT_INCOME_TOTAL','AMT_ANNUITY',"AMT_CREDIT"]
df = inp0[columns_to_keep].copy()
df





df_target_0 = df[df['TARGET'] == 0]


df_target_1 = df[df['TARGET'] == 1]

correlation_target_0 = df_target_0.corr()
correlation_target_0 = correlation_target_0.drop('TARGET', axis=0)
correlation_target_0 = correlation_target_0.drop('TARGET', axis=1)

(correlation_target_0)


correlation_target_1 = df_target_1.corr()
correlation_target_1 =correlation_target_1.drop('TARGET', axis=0)
correlation_target_1 =correlation_target_1.drop('TARGET', axis=1)
(correlation_target_1)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.heatmap(correlation_target_1, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation for clients with payment difficulties')

# Plot heatmap for all other cases
plt.subplot(1, 2, 2)
sns.heatmap(correlation_target_0, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation for all other cases')

plt.tight_layout()
plt.show()

