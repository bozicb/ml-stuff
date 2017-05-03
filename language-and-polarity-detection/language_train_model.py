import sys
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

def get_language(sentences, languages_data_folder):
   dataset = load_files(languages_data_folder)

   docs_train, docs_test, y_train, y_test = train_test_split(
      dataset.data, dataset.target, test_size=0.5)

   tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
   clf = Pipeline([('vect', tfidf_vect), ('clf', Perceptron()),])
   _ = clf.fit(dataset.data, dataset.target)

   y_predicted = clf.predict(docs_test)
   
   pickle_file = open("language_metrics.p", "wb")
   pickle.dump(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names), pickle_file)

   cm = metrics.confusion_matrix(y_test, y_predicted)
   pickle.dump(cm, pickle_file)

   #import matplotlib.pyplot as plt
   #plt.matshow(cm, cmap=plt.cm.jet)
   #plt.show()

   predicted = clf.predict(sentences)
   
   results = []
   for s, p in zip(sentences, predicted):
      results.append(dataset.target_names[p])
   return results
