import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file_path = '''
						file path that we got from EDA.
				  '''

df = pd.read_csv(input_file_path)

# Top 10 songs ever listened by counts.
print(df.groupby('title')['listen_count'].sum().sort_values(ascending = False).head(10).reset_index())

# Top 10 artists ever by listen counts.
print(df.groupby('artist_name')['listen_count'].sum().sort_values(ascending = False).head(10).reset_index())

# Top 10 releases ever by listen counts.
print(df.groupby('release')['listen_count'].sum().sort_values(ascending = False).head(10).reset_index())

# songs with more than one song ID
dic = {}
for song in df['title'].unique():
  data = df[df['title'] == song]['song_id']
  if data.nunique() > 1:
    dic[song] = list(data.unique())
print(dic)

# Lets check how one title can have more than 1 song id.
print("Lets check how one title can have more than 1 song id.")
print(dic[list(dic.keys())[0]])
print(df[df['title'] == list(dic.keys())[0]][['title','release','artist_name','year','song_id']].value_counts())

print(dic[list(dic.keys())[1]])
print(df[df['title'] == list(dic.keys())[1]][['title','release','artist_name','year','song_id']].value_counts())

print(dic[list(dic.keys())[8]])
print(df[df['title'] == list(dic.keys())[8]][['title','release','artist_name','year','song_id']].value_counts())

print(dic[list(dic.keys())[24]])
print(df[df['title'] == list(dic.keys())[24]][['title','release','artist_name','year','song_id']].value_counts())

# Conclusion
# we dont have enough data to explain why some titles have more than one song id.
# Mostly it is because it has diffrent release or artist name or year.

# Listen count over year.
sns.lineplot(df.groupby('year')['listen_count'].sum())
plt.show()

# Let's check does every song id reffers to,
# only one song (including title, artist name, release and year combined).

print(df.drop(columns=['listen_count']).duplicated().sum())


