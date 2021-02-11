import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Spotify Developer Account Credentials
cid = ''
secret = ''


def convert_json_into_df(results, sp):
    # create a list of song ids
    ids = []

    for item in results['tracks']['items']:
        track = item['track']['id']
        ids.append(track)

    song_meta = {'id': [], 'album': [], 'name': [],
                 'artist': [], 'explicit': [], 'popularity': []}

    for song_id in ids:
        # get song's meta data
        meta = sp.track(song_id)

        # song id
        song_meta['id'].append(song_id)

        # album name
        album = meta['album']['name']
        song_meta['album'] += [album]

        # song name
        song = meta['name']
        song_meta['name'] += [song]

        # artists name
        s = ', '
        artist = s.join([singer_name['name'] for singer_name in meta['artists']])
        song_meta['artist'] += [artist]

        # explicit: lyrics could be considered offensive or unsuitable for children
        explicit = meta['explicit']
        song_meta['explicit'].append(explicit)

        # song popularity
        popularity = meta['popularity']
        song_meta['popularity'].append(popularity)

    song_meta_df = pd.DataFrame.from_dict(song_meta)

    # check the song feature
    features = sp.audio_features(song_meta['id'])
    # change dictionary to dataframe
    features_df = pd.DataFrame.from_dict(features)

    # convert milliseconds to mins
    # duration_ms: The duration of the track in milliseconds.
    # 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
    features_df['duration_ms'] = features_df['duration_ms'] / 60000

    # combine two dataframe
    final_df = song_meta_df.merge(features_df)
    return final_df


def calculate_my_metrics_by_year(df_my_playlist):
    # Calculates averages of popularity, danceability, energy and valence per year of My Personal Top 100 Songs
    df_stats_per_year = df_my_playlist.groupby(['year'])['popularity', 'danceability', 'energy', 'valence'].mean()
    df_stats_per_year = df_stats_per_year.reset_index()
    df_stats_per_year.to_csv("MyTop100StatsPerYear.csv", index=False)


def get_my_2020_top50(df_my_playlist):
    # Filters my playlists data to get my personal top 50 from 2020

    # First get 2020 data, and then get top 50 resuls
    df_2020 = df_my_playlist[df_my_playlist['year'] == 2020].reset_index(drop=True)
    my_top_50 = df_2020.iloc[:50, :]
    return my_top_50


def calculate_differences_with_popular_songs(df_my_playlist, global_top_50, arg_top_50):
    # Calculates averages of popularity, danceability, energy and valence of my top 50 2020 songs, global top 50 and
    # argentina top 50

    # First, in order to have a similar number of observations, we filter our top 50
    my_top_50 = get_my_2020_top50(df_my_playlist)
    my_top_50['playlist_type'] = 'My Top 50'

    # We concat the results of the three playlists
    df_top_50s = pd.concat([my_top_50, global_top_50, arg_top_50], ignore_index=True)

    # We groupby playlist type and get results for the same variables
    df_top_50s_stats = df_top_50s.groupby(['playlist_type'])['popularity', 'danceability', 'energy', 'valence'].mean()
    df_top_50s_stats = df_top_50s_stats.reset_index()
    df_top_50s_stats.to_csv("MyTop50vsGlobalAndArg.csv", index=False, encoding='utf-8 sig')

    # Additionally, we carry out a dimensionality reduction with TNSE and plot the different songs
    # However, before TNSE, we scale data
    scaler = StandardScaler()
    df_top_50_scaled = scaler.fit_transform(df_top_50s[['popularity', 'danceability', 'energy', 'key', 'loudness',
                                                        'mode', 'speechiness', 'acousticness', 'instrumentalness',
                                                        'liveness', 'valence', 'tempo']].values)
    # We carry out TNSE and save everything in a dataframe
    tnse_features = TSNE(n_components=2).fit_transform(df_top_50_scaled)
    projection = pd.DataFrame(data={
        'x': tnse_features[:, 0],
        'y': tnse_features[:, 1],
        'playlist_type': df_top_50s['playlist_type']
    })
    # We plot the TNSE
    fig = px.scatter(
        projection, x='x', y='y', color='playlist_type', title='Resultado del TNSE por Playlist')
    fig.update_traces(marker=dict(size=12))
    fig.write_image('tnse.png')


def turn_rows_into_vectors(df_my_playlist):
    # Converts the songs of my playlist into numpy arrays and appends it into a list
    my_playlist_np = {}
    ix = 0
    for index, row in df_my_playlist.iterrows():
        my_playlist_np[ix] = np.array((row['danceability'], row['energy'], row['key'],
                                        row['loudness'], row['acousticness'], row['instrumentalness'],
                                        row['liveness'], row['valence']))
        ix += 1
    return my_playlist_np


def calculate_distance_with_my_playlist(danceability, energy, key, loudness, acousticness, instrumentalness,
                                        liveness, valence, my_playlist_np):
    # Function that calculates total distance between a song in the Kaggle dataset and the 500 songs of my playlist

    # Initialize total distance and create array
    total_ditance = 0
    song_array = np.array((danceability, energy, key, loudness, acousticness, instrumentalness, liveness, valence))

    # Iterate and calculate total distance
    for song in my_playlist_np:
        total_ditance += np.linalg.norm(song_array - my_playlist_np[song])

    return total_ditance


def simple_recommender(df_my_playlist, df_kaggle, n_songs):
    # Function that tries to recommend new songs to here, calculating total euclidian distance between my playlists
    # and each song

    # First we turn our playlist into a numpy array, calculating the average of each variable
    my_playlist_np = {0: np.array((np.mean(df_my_playlist['danceability']), np.mean(df_my_playlist['energy']),
                                   np.mean(df_my_playlist['key']), np.mean(df_my_playlist['loudness']),
                                   np.mean(df_my_playlist['acousticness']), np.mean(df_my_playlist['instrumentalness']),
                                   np.mean(df_my_playlist['liveness']), np.mean(df_my_playlist['valence'])))}

    # Calculate the distance of each song with our playlist
    df_kaggle['distance'] = np.vectorize(calculate_distance_with_my_playlist)(df_kaggle['danceability'],
                                                                              df_kaggle['energy'], df_kaggle['key'],
                                                                              df_kaggle['loudness'],
                                                                              df_kaggle['acousticness'],
                                                                              df_kaggle['instrumentalness'],
                                                                              df_kaggle['liveness'],
                                                                              df_kaggle['valence'], my_playlist_np)
    # Order by distance
    df_kaggle = df_kaggle.sort_values(by=['distance']).reset_index(drop=True)
    print(df_kaggle[['artist', 'name']].head(n_songs))
    df_kaggle.head(n_songs).to_csv("average_recomendation.csv", index=False, encoding='utf-8 sig')


def main():
    # Connect to API
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Download results from Top 100 songs from each year
    playlist_id_dict = {
        2016: '',
        2017: '',
        2018: '',
        2019: '',
        2020: ''
    }

    # Get info of the yearly playlist and concat them into a single data frame
    df_my_playlist = pd.DataFrame()
    for i in range(2016, 2021):
        df_year = convert_json_into_df(results=sp.playlist(playlist_id_dict[i]), sp=sp)
        df_year['year'] = i
        df_my_playlist = pd.concat([df_my_playlist, df_year], ignore_index=True)
    calculate_my_metrics_by_year(df_my_playlist)

    # Download results of Global Top 50 and Argentina Top 50
    global_top_50 = convert_json_into_df(results=sp.playlist(''), sp=sp)
    global_top_50['playlist_type'] = 'Global Top 50'

    arg_top_50 = convert_json_into_df(results=sp.playlist(''), sp=sp)
    arg_top_50['playlist_type'] = 'Arg Top 50'

    # Calculate differences with popular songs
    calculate_differences_with_popular_songs(df_my_playlist, global_top_50, arg_top_50)

    # Calculate recommended songs from Kaggle dataset
    df_kaggle = pd.read_csv("kaggle_spotify_data.csv")
    df_kaggle.rename(columns={'artists': 'artist'}, inplace=True)

    # Run recommender
    simple_recommender(df_my_playlist, df_kaggle, 25)


if __name__ == '__main__':
    main()
