import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file_path = '''
					path to file
				 '''

df = pd.read_csv(data_file_path,sep=';')

print(df.sample(3).T)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# number of unique categories for every categorical feature
ls = df.drop(['listen_count','year'],axis = 1).columns
for col in ls:
  print(f'{col} : {round(df[col].nunique()/df.shape[0]*100,2)} % : {df[col].nunique()}')

# ploting distplot and boxplot for nummerical features
for col in ['listen_count', 'year']:
  plt.figure(figsize = (10,5))
  plt.subplot(2,1,1)
  sns.histplot(df[col])
  plt.xlabel('')
  plt.subplot(2,1,2)
  sns.boxplot(df[col],orient = 'h')
  plt.show()
  print('')

# too much of skewness
print(df['listen_count'].skew())

# function for calculting upper limit and lower limit
def upper_lower_limits(col):
  if type(df[col][0]) == np.int64:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    return upper, lower

# number of rows that are out of statistical limits
print(np.where(df['listen_count'] > upper_lower_limits("listen_count")[0])[0].shape[0])
# % of number of rows that are out of statistical limits
print(f"{np.where(df['listen_count'] > upper_lower_limits('listen_count')[0])[0].shape[0]/df.shape[0]*100} % ")
# lower limit is -ve, no need to check


# only 8.883 % are out of statistical limit
# two ways
# make them null and impute
# or
# replace them by upper limit

# we accept second because they are statistically wrong but can be right as
# a person likes to listen some songs 2-3 times a day for a week or two.
# probability of that is not zero hence we give them max statistical limit which is 6.0

upper_limit = upper_lower_limits("listen_count")[0]
df['listen_count'] = np.where(df['listen_count'] > upper_limit, upper_limit ,df['listen_count']).\
astype(np.int64)

plt.figure(figsize = (10,5))
plt.subplot(2,1,1)
sns.distplot(df['listen_count'])
plt.xlabel('')
plt.subplot(2,1,2)
sns.boxplot(df['listen_count'],orient = 'h')
plt.show()

# here only 0 is out of statistical limit
print(f"{round(np.where(df['year'] == 0)[0].shape[0]/df.shape[0]*100,2)} %")

# 18% of values are 0
# here we will extract data if there is any song that has multiplt rows with year of release as 0

songs_zero_year = list(df[df['year'] == 0]['title'])

dic ={}
for song in songs_zero_year:
  n = df[df['title'] == song]['year'].unique()
  if n.shape[0] > 1:
    dic[song] = n
print(len(dic))

# difference in year is due to difference in singer. Those 73 titles are coincidentally same as other.
df['year'] = np.where(df['year'] == 0, np.nan, df['year'])
df['year'].iloc[np.random.randint(df.dropna().shape[0])]

print(df.isnull().sum())

# Random imputation
df['year'][df['year'].isnull()] = df['year'].dropna().sample(df['year'].isnull().sum()).values
df['year'] = df['year'].astype(np.int64)
print(df.isnull().sum())

print(df.info())

# ploting after random imputation of value 0 in feature 'year'.
plt.figure(figsize = (10,5))
plt.subplot(2,1,1)
sns.distplot(df['year'])
plt.xlabel('')
plt.subplot(2,1,2)
sns.boxplot(df['year'],orient = 'h')
plt.show()

print(upper_lower_limits('year'))

upper_limit = upper_lower_limits('year')[0]
lower_limit = upper_lower_limits('year')[1]

df['year'] = np.where(df['year'] < lower_limit, lower_limit, df['year'])
df['year'] = df['year'].astype(np.int64)

plt.figure(figsize = (10,5))
plt.subplot(2,1,1)
sns.distplot(df['year'])
plt.xlabel('')
plt.subplot(2,1,2)
sns.boxplot(df['year'],orient = 'h')
plt.show()

output_file_path = '''
					path to file
				 '''

df.to_csv(output_file_path)
