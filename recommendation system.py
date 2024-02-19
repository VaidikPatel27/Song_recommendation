import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

input_file_path = '''
						file path that we got from EDA.
				  '''

df = pd.read_csv(input_file_path)

print(df.sample(5))

final_df = pd.pivot_table(df,
                          columns = 'song_id',
                          index = 'user_id',
                          values = 'listen_count')

final_df.fillna(0,
                inplace = True)

final_df = final_df.astype(np.int64)

print(final_df.sample(5))

similarity_matrix = cosine_similarity(final_df)

def remove_element(array1, array2):
  for element in array2:
    if element in array1:
      array1.remove(element)
  return array1

def collabrative_recommendation(artist, songs = 10, full_info = False):

  # to get the index number of artist that we got
  artist_arr = final_df.index == artist
  artist_idx = np.where(artist_arr == True)[0][0]

  # cosine similarity to get recommended index of song
  recommended_idx = np.argsort(similarity_matrix[artist_idx])

  # find indexes of song where users already gave review or rating
  song_idx = np.where(final_df.iloc[artist_idx,:] > 0)[0]

  # create a variable to store all the indexes where song listen count is zero.
  final_recommendation = np.array(remove_element(recommended_idx.tolist(), song_idx.tolist()))[1:songs+1]

  # coverting index into song id
  song_list = list(final_df.iloc[:,final_recommendation].columns)

  # coverting index into song name
  song_title = []
  for song_id in song_list:
    song_title.append(df[df['song_id'] == song_id]['title'].unique()[0])

  # program to get full info of song
  if full_info == True:
    song_title = []
    for song_id in song_list:
      details = {}
      details['title'] = df[df['song_id'] == song_id]['title'].unique()[0]
      details['artist_name'] = df[df['song_id'] == song_id]['artist_name'].unique()[0]
      details['release'] = df[df['song_id'] == song_id]['release'].unique()[0]
      details['year'] = df[df['song_id'] == song_id]['year'].unique()[0]
      song_title.append(details)

  else:
    song_title = []
    for song_id in song_list:
      song_title.append(df[df['song_id'] == song_id]['title'].unique()[0])

  return song_title




# Testing

recommended_songs = collabrative_recommendation(df['user_id'].sample().iloc[0],
                            songs = 10,
                            full_info = False)

print(recommended_songs)


