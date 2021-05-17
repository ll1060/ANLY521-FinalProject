import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
import gensim
import argparse
from sklearn import naive_bayes

def main(yelp_reivew):
    #read in the dataset
    yelp_reviews = pd.read_csv('yelp_review.csv')


    # filter the data to contain only <=2 star and >=4 star reviews
    df_12_45_star = yelp_reviews[(yelp_reviews['stars'] == 1.0) |
                               (yelp_reviews['stars'] == 2.0) |
                               (yelp_reviews['stars'] == 4.0) |
                               (yelp_reviews['stars'] == 5.0)]

    # create a new column in the df_12_45_star to make <=2 star and >= 4 star ratings into 2 classes
    row_indexes_45 = df_12_45_star[df_12_45_star['stars'] >= 4.0].index
    row_indexes_12 = df_12_45_star[df_12_45_star['stars'] <= 2.0].index
    # create a new column named 'star class' and classify reviews with >= 4 stars as good, <= 2 stars as bad
    df_12_45_star.loc[row_indexes_45, 'star_class'] = "good"
    df_12_45_star.loc[row_indexes_12, 'star_class'] = "bad"

    # encode the "star class" column
    le = preprocessing.LabelEncoder()
    le.fit(df_12_45_star.star_class)
    df_12_45_star['label'] = le.transform(df_12_45_star.star_class)


    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df_12_45_star['text'], df_12_45_star['label'],
                                                        random_state = 0,test_size=0.25)

    # initiate a tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       ngram_range=(1, 2),
                                       lowercase=True,
                                       max_features=3000)
    # transform X_train and X_test
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # train and test a naive bayes with tfidf vectors
    nb_tfidf = naive_bayes.MultinomialNB()
    nb_tfidf.fit(X_train_tfidf,y_train)
    # predict the labels on test dataset
    predictions_nb = nb_tfidf.predict(X_test_tfidf)
    # check precision, recall, f1-score for naive bayes with tfidf
    print("Naive Bayes with TFIDF vectors classification report:\n", metrics.classification_report(y_test,predictions_nb))

    # initiate logistic regression model with tfidf vectors, fit the model to train set and test the model on test set
    logreg_tfidf = LogisticRegression(random_state=0).fit(X_train_tfidf, y_train)
    logreg_tfidf_pred = logreg_tfidf.predict(X_test_tfidf)

    # check precision, recall, f1-score for logistic regression with tfidf
    print("Logistic Regression with Tfidf Vectors classification report:\n",
          metrics.classification_report(y_test, logreg_tfidf_pred))

    # train and test a linear SVM with tfidf vectors
    svm_tfidf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
    svm_tfidf.fit(X_train_tfidf, y_train)
    # predict the labels on validation dataset
    svm_tfidf_pred = svm_tfidf.predict(X_test_tfidf)
    # check precision, recall, f1-score for linear SVM with tfidf
    print("Linear SVM with TFIDF vectors classification report:\n",
          metrics.classification_report(y_test, svm_tfidf_pred))

    # build word embedding vectors
    # initiate a word2vec model
    w2v = gensim.models.Word2Vec(list(df_12_45_star.text), size=200, window=10, min_count=2, iter=10)

    # write a function to calculate the mean of word2vec vectors
    def document_vector(doc):
        """Create document vectors by averaging word vectors. Also remove out-of-vocabulary words."""
        doc = [word for word in doc if word in w2v.wv.vocab]
        return np.mean(w2v[doc], axis=0)

    # build w2v vectors for train and test data
    X_train['doc_vector'] = X_train.apply(document_vector)
    X_test['doc_vector'] = X_test.apply(document_vector)
    X_train_w2v = list(X_train['doc_vector'])
    X_test_w2v = list(X_test['doc_vector'])

    # train a logistic regression with w2v vectors
    logreg_w2v = LogisticRegression(random_state=0).fit(X_train_w2v, y_train)
    # make predictions using the model trained on test set
    logreg_w2v_pred = logreg_w2v.predict(X_test_w2v)
    print("Logistic Regression with w2v vectors classification report:\n",
          metrics.classification_report(y_test, logreg_w2v_pred))

    # train a linear SVM with w2v vectors on the train set
    svm_w2v = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_w2v.fit(X_train_w2v, y_train)
    # make predictions using the trained model on test set
    svm_w2v_pred = svm_w2v.predict(X_test_w2v)
    print("Linear SVM with w2v classification report:\n", metrics.classification_report(y_test, svm_w2v_pred))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp_review", type=str, default="yelp_review.csv",
                        help="yelp review file")
    args = parser.parse_args()
    main(args.yelp_review)