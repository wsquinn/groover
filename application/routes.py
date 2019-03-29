from flask import render_template, flash, redirect
from application import app
from application.forms import LoginForm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import urllib.request
from bs4 import BeautifulSoup
import csv
import requests
import json
import os

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'World'}
    return render_template('index.html', title='Home', user=user)

@app.route('/lookup', methods=['GET', 'POST'])
def lookup():
    form = LoginForm()
    if form.validate_on_submit():
        return redirect('/recommendations/' + form.artist.data + '/' +  form.title.data)
    return render_template('lookup.html', title='Look up song', form=form)

@app.route('/recommendations/<artist>/<title>')
def recommendations(artist, title):
    #TODO: store this in a file so we don't have to do this over and over
    #TODO: add album art to each track in this csv
    docLabels = []
    with open('data/songdata.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                docLabels.append(row[1] + " ~ " + row[0])
                if len(docLabels) % 100 == 0:
                    print("\rReading documents: %d" % len(docLabels), end='', flush=True)
            line_count += 1

    url = 'matcher.track.get?q_track={}&q_artist={}&format={}'.format(title, artist,'json')
    req_url = 'http://api.musixmatch.com/ws/1.1/{}&apikey={}'.format(url, os.environ.get("MUSIX_API_KEY"))
    data = requests.get(req_url)
    data = json.loads(data.text)
    if data["message"]["header"]["status_code"] == 200:

        matched_artist = data["message"]["body"]["track"]["artist_name"]
        matched_title = data["message"]["body"]["track"]["track_name"]

        flash('requested song: {} ~ {}'.format(
            matched_artist, matched_title))

        matched_artist = matched_artist.lower()
        matched_title = matched_title.lower()

        # remove all except alphanumeric characters from matched_artist and matched_title
        matched_artist = re.sub('[^A-Za-z0-9]+', "", matched_artist)
        matched_title = re.sub('[^A-Za-z0-9]+', "", matched_title)

        if matched_artist.startswith("the"):    # remove starting 'the' from matched_artist e.g. the who -> who
           matched_artist = matched_artist[3:]
        url = "http://azlyrics.com/lyrics/"+matched_artist+"/"+matched_title+".html"

        try:
           content = urllib.request.urlopen(url).read()
           soup = BeautifulSoup(content, 'html.parser')
           lyrics = str(soup)
           # lyrics lies between up_partition and down_partition
           up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
           down_partition = '<!-- MxM banner -->'
           lyrics = lyrics.split(up_partition)[1]
           lyrics = lyrics.split(down_partition)[0]
           lyrics = lyrics.replace('</div>','').strip()
           lyrics = lyrics.replace('/','').replace('<br>','').replace('\n',' ').replace('<i>','').replace('\'','').replace('  ','').replace('"','')
           lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)

        except Exception as e:
           return "Exception occurred \n" +str(e)

        lyrics = lyrics.strip()
        lyrics = lyrics.replace('\n',' ')
        lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)

        model= Doc2Vec.load("d2v.model")
        #to find the vector of a document which is not in training data
        test_data = word_tokenize(lyrics.lower())
        v1 = model.infer_vector(doc_words=test_data, alpha=0.025, min_alpha=0.001, steps=55)
        similar_v1 = model.docvecs.most_similar(positive=[v1])

        for song in similar_v1:
            flash('recommended song: {}'.format(
                docLabels[int(song[0])]
                ))
    else:
        flash('Sorry, we did not find the track "{}" by {}. Try again?'.format(
            title, artist))

    return render_template('recommendations.html', title='Your recommendations')
