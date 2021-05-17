# ANLY521-FinalProject

This project aims to shed light on the business’s ratings on Yelp using user’s textual reviews
and built classifiers to predict the star ratings. Furthermore, this project will investigate 
the ability for each model developed to distinguish between 1-star and 5-star ratings and below 
2-star and above 4-star ratings. 

# Files
1. yelp_review.csv: dataset used in this project.
   
2. yelp_1_5_star_Review_Classification.py: contains everything in classification task 1: classify 
1-star and 5-star ratings. filter dataset so that only reviews with 1-star and 5-star are remained,
train and test split, initialize tf-idf vectorizer, train word2vec model, calculate average of word2vec, 
   train and test and metrics for model performance. 
   
example usage: python yelp_1_5_star_Review_Classification.py --yelp_review yelp_review.csv

3. yelp_2_4_star_Review_Classification.py: contains everything in classification task 2: classify 
<=2-star and >=4-star ratings. Filter dataset so that only reviews with <=2-star and >=4-star are remained,
train and test split, initialize tf-idf vectorizer, train word2vec model, calculate average of word2vec, 
   train and test and metrics for model performance. 
   
example usage: python yelp_2_4_star_Review_Classification.py --yelp_review yelp_review.csv

# Requirements:
The follow packages are needed:\
os\
numpy\
pandas\
sklearn.preprocessing\
sklearn.preprocessing: LabelEncoder\
sklearn.linear_model: LogisticRegression\
sklearn.feature_extraction.text: TfidfVectorizer\
sklearn.model_selection: train_test_split\
smart_open\
gensim.models:Word2Vec\
sklearn: metrics\
sklearn: svm\
gensim\
argparse\
sklearn: naive_bayes\
