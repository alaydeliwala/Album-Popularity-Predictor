from bs4 import BeautifulSoup
import requests
import csv


class Album:
    name = ""
    ranking = 0


# Removes all foramting from a string
def clean_data(data):
    temp = data.replace('\r', '').replace('\n', '').replace(',', '').strip()
    return temp


def get_billboard_top_200_albums(year, f):
    missing_album_ranks = []
    for x in range(1, 201):
        missing_album_ranks.append(int(x))

    url = 'https://www.billboard.com/charts/year-end/' + str(
        year) + '/top-billboard-200-albums'
    response = requests.get(url)
    bs_doc = BeautifulSoup(response.text, 'lxml')
    rows = bs_doc.find_all("article")
    # print(len(rows))
    albums = []
    for item in rows:
        rank = int(item.find(class_='ye-chart-item__rank').get_text())
        title = clean_data(
            str(item.find(class_='ye-chart-item__title').get_text()))
        artist = clean_data(
            str(item.find(class_='ye-chart-item__artist').get_text()))
        missing_album_ranks.remove(rank)
        # print(title + ' ' + artist + ' ' + str(rank))
        f.writerow([rank, title, artist, year])
    print('Missing entries for ' + str(year), missing_album_ranks)


def main():
    with open('data/billboard_album_data-missing.csv', 'w', newline='') as f:
        filewriter = csv.writer(f,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['rank', 'album_name', 'artist', 'year'])
        for x in range(2008, 2019):
            get_billboard_top_200_albums(x, filewriter)


if (__name__ == "__main__"):
    main()
