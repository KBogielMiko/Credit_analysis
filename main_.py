'''1. EDA'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set_style('darkgrid')

data = pd.read_csv("german_data_credit_cat.csv")
df = pd.DataFrame(data)


Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 DM", 'A12': "0 <= <200 DM",'A13':">= 200 DM "}
df["Status of existing checking account"]=df["Status of existing checking account"].map(Status_of_existing_checking_account)

Credit_history={"A34":"critical account","A33":"delay in paying off","A32":"existing credits paid back duly till now","A31":"all credits at this bank paid back duly","A30":"no credits taken"}
df["Credit history"]=df["Credit history"].map(Credit_history)

Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "furniture/equipment", "A43" :"radio/television" , "A44" : "domestic appliances", "A45" : "repairs", "A46" : "education", 'A47' : 'vacation','A48' : 'retraining','A49' : 'business','A410' : 'others'}
df["Purpose"]=df["Purpose"].map(Purpose)

Saving_account={"A65" : "no savings account","A61" :"<100 DM","A62" : "100 <= <500 DM","A63" :"500 <= < 1000 DM", "A64" :">= 1000 DM"}
df["Savings account/bonds"]=df["Savings account/bonds"].map(Saving_account)

Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"unemployed"}
df["Present employment since"]=df["Present employment since"].map(Present_employment)

Personal_status_and_sex={ 'A95':"female:single",'A94':"male:married/widowed",'A93':"male:single", 'A92':"female:divorced/separated/married", 'A91':"male:divorced/separated"}
df["Personal status and sex"]=df["Personal status and sex"].map(Personal_status_and_sex)

Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"}
df["Other debtors / guarantors"]=df["Other debtors / guarantors"].map(Other_debtors_guarantors)

Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}
df["Property"]=df["Property"].map(Property)

Other_installment_plans={'A143':"none", 'A142':"store", 'A141':"bank"}
df["Other installment plans"]=df["Other installment plans"].map(Other_installment_plans)

Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}
df["Housing"]=df["Housing"].map(Housing)

Job={'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"}
df["Job"]=df["Job"].map(Job)

Telephone={'A192':"yes", 'A191':"none"}
df["Telephone"]=df["Telephone"].map(Telephone)

foreign_worker={'A201':"yes", 'A202':"no"}
df["foreign worker"]=df["foreign worker"].map(foreign_worker)


df.head()
df.info()
df.shape
df.dtypes
df.tail()
df.sample(5)

# no missings

#visualisation
df.hist(figsize=(10,10), xrot=45)
plt.show()

df.describe()
df.describe(include='object')
#no outliers

df['Cost Matrix(Risk)']

for col in df.select_dtypes(include='object'):
    if df[col].nunique() <= 22:
        sns.countplot(y=col, data=df)
        plt.show()

# We dont have any variable which we can replace by float - no any margin string to covert to floats
num_cols = ['Duration in month','Credit amount', 'Installment rate in percentage of disposable income', 'Present residence since','Age in years','Number of existing credits at this bank','Number of people being liable to provide maintenance for']
for col in num_cols:
    sns.boxplot(y = df['Cost Matrix(Risk)'].astype('category'), x = col, data=df)
    plt.show()

for col in df.select_dtypes(include='object'):
    if df[col].nunique() <=5:
        display(pd.crosstab(df['Cost Matrix(Risk)'], df[col], normalize='index'))

for col in df.select_dtypes(include='object'):
    if df[col].nunique() <= 5:
        g = sns.catplot(x = col, kind='count', col = 'Cost Matrix(Risk)', data=df, sharey=False)
        g.set_xticklabels(rotation=60)

for col in df.select_dtypes(include='object'):
    if df[col].nunique() <= 5:
        display(df.groupby(col)[['Duration in month','Credit amount', 'Installment rate in percentage of disposable income', 'Present residence since','Age in years','Number of existing credits at this bank','Number of people being liable to provide maintenance for']].mean())

corr = data.corr()
corr

plt.figure(figsize=(6,6))
sns.heatmap(corr, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
plt.show()


#SVM algoritm
#Firstly I will try to check efficiency of SVM algorithm (algorithm should work quite well with relatively small database and relatively many features)

#Split into train/test sets

X_train, X_test, y_train, y_test = train_test_split(df, df["Cost Matrix(Risk)"], test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape)

y_test

target = df["Cost Matrix(Risk)"]
num_features = df.select_dtypes(include=['float64','int64']).columns
cat_features = df.select_dtypes(include=['object']).columns

num_features = ['Duration in month', 'Credit amount',
       'Installment rate in percentage of disposable income',
       'Present residence since', 'Age in years',
       'Number of existing credits at this bank',
       'Number of people being liable to provide maintenance for']

#OneHot Encoding

dummLev = pd.get_dummies(df[cat_features], drop_first=True)
df_ = pd.concat([df[num_features], dummLev, df["Cost Matrix(Risk)"]], axis=1)
df_.columns

df[num_features] = df[num_features].apply(lambda x: (x-x.mean())/x.std())
features = df_.columns.tolist()
features.remove('Cost Matrix(Risk)')
print(features)
