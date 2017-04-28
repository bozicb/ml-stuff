import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LinearSVC())]) 
    text_clf = text_clf.fit(dataset.data, dataset.target)
    parameters = {'vect__ngram_range': [(1,1), (1,2)]}
    gs_clf = GridSearchCV(text_clf, parameters)
    gs_clf = gs_clf.fit(dataset.data, dataset.target)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    y_predicted = gs_clf.predict(docs_test)
    print('Accuracy', np.mean(y_predicted == dataset.target))

    print(gs_clf.cv_results_)

    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    import matplotlib.pyplot as plt
    plt.matshow(cm)
    plt.show()
