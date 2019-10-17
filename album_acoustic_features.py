from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('data/acoustic_features.csv',
        usecols =['song',
        'album',
        'artist',
        'acousticness',
        'danceability',
        'energy',
        'instrumentalness',
        'liveness',
        'loudness',
        'speechiness',
        'tempo',
        'date'])
    df.to_csv('data/album_acoustic_features.csv')
    df = pd.read_csv('data/album_acoustic_features.csv')
    mask = (df['date'] > "2009-01-01") & (df.shape[1] == 13)
    df[mask].groupby('album').agg(
        acousticness_mean=('acousticness', np.mean),
        acousticness_var=('acousticness', np.var),
        danceability_mean=('danceability', np.mean),
        danceability_var=('danceability', np.var),
        energy_mean=('energy', np.mean),
        energy_var=('energy', np.var),
        instrumentalness_mean=('instrumentalness', np.mean),
        instrumentalness_var=('instrumentalness', np.var),
        liveness_mean=('liveness', np.mean),
        liveness_var=('liveness', np.var),
        loudness_mean=('loudness', np.mean),
        loudness_var=('loudness', np.var),
        speechiness_mean=('speechiness', np.mean),
        speechiness_var=('speechiness', np.var),
        tempo_mean=('tempo', np.mean),
        tempo_var=('tempo', np.var)
     ).to_csv('data/album_acoustic_features.csv')

if (__name__ == "__main__"):
    main()
