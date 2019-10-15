from bs4 import BeautifulSoup
import requests
import csv


def get_billboard_top_200_albums():
    albums = {}
    with open('data/acoustic_features.csv', newline='') as csvfile:
        album_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(album_reader)
        for row in album_reader:
            if int(row[-1][:4]) >= 2008:
                if (albums.get(row[2])):
                    albums.update({row[2]: albums.get(row[2]).append(row)})
                else:
                    temp = []
                    temp.append(row)
                    albums.update({row[2]: temp})

            # print(row)
    print(albums)


def main():
    # reduce_data(1999, 2018)
    with open('data/complied_album_features.csv', 'w', newline='') as f:
        filewriter = csv.writer(f,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([
            'album',
            'artist',
            'acousticness_mean',
            'acousticness_var',
            'danceability_mean',
            'danceability_var',
            'energy_mean',
            'energy_var',
            'instrumentalness_mean',
            'instrumentalness_var',
            'liveness_mean',
            'liveness_var',
            'loudness_mean',
            'loudness_var',
            'speechiness_mean',
            'speechiness_var',
            'tempo_mean',
            'tempo_var',
            'year',
        ])
        get_billboard_top_200_albums()


if (__name__ == "__main__"):
    main()
