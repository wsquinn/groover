#       azapi.py, an api for AZLyrics
#
#       Copyright 2019 Khaled El-Morshedy <elmoiv>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License 3 as published by
#       the Free Software Foundation.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import requests
from lxml import html

class AZlyric:
    def __init__(self, Title, Artist):
        self.title = Title.replace(' ','').lower()
        self.artist = Artist.replace(' ','').lower()

    def Get(self, save = False):
        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
        page = requests.get(f'http://www.azlyrics.com/lyrics/{self.artist}/{self.title}.html', headers=headers)
        tree = html.fromstring(page.content)
        data = tree.xpath('/html/body/div[3]/div/div[2]/div[5]//text()')
        lyrics = ''.join(data[1:])
        if save and lyrics != '':
            with open(self.title.title() + ' - ' + self.artist.title() + '.txt', 'w') as lrc:
                lrc.write(lyrics)
        return lyrics
