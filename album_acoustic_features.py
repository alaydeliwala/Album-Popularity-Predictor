from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import numpy as np

#def get_billboard_top_200_albums(filewriter):
#    albums = {}
#
#    with open('data/acoustic_features.csv', newline='') as csvfile:
#        album_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#        #save the labels
#        header = next(album_reader)
#        for row in album_reader:
#            if int(row[-1][:4]) >= 2008:
#                if (albums.get(row[2])):
#                    albums.update({row[2]: albums.get(row[2]).append(row)})
#                else:
#                    temp = []
#                    temp.append(row)
#                    albums.update({row[2]: temp})
        # look through all the albums
        # for entry in albums:
        
            # use pandas to calculate mean an variance of each attribute listed below
            # m an entry into the csv with the follwing info
            # print(row)
    ##print(albums)

def main():
    # reduce_data(1999, 2018)
#    with open('data/complied_album_features.csv', 'w', newline='') as file:
#        filewriter = csv.writer(file,
#                                delimiter=',',
#                                quotechar='|',
#                                quoting=csv.QUOTE_MINIMAL)
#        filewriter.writerow([
#            'album',
#            'artist',
#            'acousticness_mean',
#            'acousticness_var',
#            'danceability_mean',
#            'danceability_var',
#            'energy_mean',
#            'energy_var',
#            'instrumentalness_mean',
#            'instrumentalness_var',
#            'liveness_mean',
#            'liveness_var',
#            'loudness_mean',
#            'loudness_var',
#            'speechiness_mean',
#            'speechiness_var',
#            'tempo_mean',
#            'tempo_var',
#            'year',
#        ])
#        get_billboard_top_200_albums(filewriter)

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
    mask = (df['date'] > "2008-01-01") & (df.shape[1] == 13)
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
#    df['acousticness_mean'] = df['acousticness'].mean()
#    df['acousticness_var'] = df['acousticness'].var()
#    df['danceability_mean'] = df['danceability'].mean()
#    df['danceability_var'] = df['danceability'].var()
#    df['energy_mean'] = df['energy'].mean()
#    df['energy_var'] = df['energy'].var()
#    df['instrumentalness_mean'] = df['instrumentalness'].mean()
#    df['instrumentalness_var'] = df['instrumentalness'].var()
#    df['liveness_mean'] = df['liveness'].mean()
#    df['liveness_var'] = df['liveness'].var()
#    df['loudness_mean'] = df['loudness'].mean()
#    df['loudness_var'] = df['loudness'].var()
#    df['speechiness_mean'] = df['speechiness'].mean()
#    df['speechiness_var'] = df['speechiness'].var()
#    df['tempo_mean'] = df['tempo'].mean()
#    df['tempo_var'] = df['tempo'].var()

if (__name__ == "__main__"):
    main()
