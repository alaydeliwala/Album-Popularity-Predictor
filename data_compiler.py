import pandas as pd



def main():
    album_df  = pd.read_csv('data/billboard_album_data-missing.csv')
    album_features_df  = pd.read_csv('data/album_acoustic_features.csv')
    for album in album_df['album_name']:
        print(album)

if (__name__ == "__main__"):
    main()