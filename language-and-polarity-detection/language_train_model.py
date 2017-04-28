import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


languages_data_folder = sys.argv[1]
dataset = load_files(languages_data_folder)

docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
clf = Pipeline([('vect', tfidf_vect), ('clf', Perceptron()),])
_ = clf.fit(dataset.data, dataset.target)

y_predicted = clf.predict(docs_test)

print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))

cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

import matplotlib.pyplot as plt
plt.matshow(cm, cmap=plt.cm.jet)
plt.show()

sentences = [
    u'This is a language detection test.',
    u'Ceci est un test de d\xe9tection de la langue.',
    u'Dies ist ein Test, um die Sprache zu erkennen.',
]
predicted = clf.predict(sentences)

for s, p in zip(sentences, predicted):
    print(u'The language of "%s" is "%s"' % (s, dataset.target_names[p]))
