#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv

# #create a list data that stores the content of all text files in order of their names in docLabels
data = []
docLabels = []
with open('data/songdata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            data.append(row[3])
            docLabels.append(row[1] + " ~ " + row[0])
            if len(data) % 100 == 0:
                print("\rReading documents: %d" % len(data), end='', flush=True)
        line_count += 1

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 4
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("hello".lower())
v1 = model.infer_vector(doc_words=test_data, alpha=0.025, min_alpha=0.001, steps=55)
similar_v1 = model.docvecs.most_similar(positive=[v1])
print("V1_infer", v1)
print(similar_v1)

for song in similar_v1:
    print(docLabels[int(song[0])])

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
for song in similar_doc:
    print(docLabels[int(song[0])])
print("similar doc")
print(similar_doc)
