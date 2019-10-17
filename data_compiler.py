import pandas as pd



def main():
    album_df = pd.read_csv('data/billboard_album_data.csv')
    album_features_df  = pd.read_csv('data/album_acoustic_features.csv')
    output_df = pd.DataFrame(columns=['album', 'artist','year','acousticness_mean','acousticness_var','danceability_mean','danceability_var','energy_mean','energy_var','instrumentalness_mean','instrumentalness_var','liveness_mean','liveness_var','loudness_mean','loudness_var','speechiness_mean','speechiness_var','tempo_mean','tempo_var','rank'])
    match_count=0
    for album,artist,year,rank in zip(album_df['album_name'],album_df['artist'],album_df['year'],album_df['rank']):
        if album_features_df.loc[album_features_df['album'] == album].size == 0:
            pass
        else:
            temp_df = [
                album,
                artist,
                year,
                album_features_df.loc[album_features_df['album'] == album].values[0][1],
                album_features_df.loc[album_features_df['album'] == album].values[0][2],
                album_features_df.loc[album_features_df['album'] == album].values[0][3],
                album_features_df.loc[album_features_df['album'] == album].values[0][4],
                album_features_df.loc[album_features_df['album'] == album].values[0][5],
                album_features_df.loc[album_features_df['album'] == album].values[0][6],
                album_features_df.loc[album_features_df['album'] == album].values[0][7],
                album_features_df.loc[album_features_df['album'] == album].values[0][8],
                album_features_df.loc[album_features_df['album'] == album].values[0][9],
                album_features_df.loc[album_features_df['album'] == album].values[0][10],
                album_features_df.loc[album_features_df['album'] == album].values[0][11],
                album_features_df.loc[album_features_df['album'] == album].values[0][12],
                album_features_df.loc[album_features_df['album'] == album].values[0][13],
                album_features_df.loc[album_features_df['album'] == album].values[0][14],
                album_features_df.loc[album_features_df['album'] == album].values[0][15],
                album_features_df.loc[album_features_df['album'] == album].values[0][16],
                rank
                ]

            output_df.loc[match_count] = temp_df
            match_count += 1
        
    print("Number of entries that found a match: " + str(match_count))
    output_df.to_csv('data/full_album_data.csv',index=False)

    
if (__name__ == "__main__"):
    main()