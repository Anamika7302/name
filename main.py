
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score , classification_report

# loading the data
data = pd.read_csv("news1.csv", dtype=str)
data = data.loc[:, [not str(col).startswith('Unnamed') for col in data.columns]]


# cleaning the text and title
data["title"]=data['title'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
data['text'] = data['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

data=data.drop("label",axis=1)

# print(data.columns)

# combining the text and title to train my model well
data["combined_text"]=data["title"] +" " + data["text"]

# Specifiying features and labels
X =data["combined_text"]
Y =data["labels"]


#  Spliting my data
split =StratifiedShuffleSplit(n_splits= 1, test_size=0.2,random_state=42)

for train_index , test_index in split.split(data, data["labels"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
    
#  redifining the  features and labels after splitting for training and testing
X_train=strat_train_set["combined_text"]
X_test=strat_test_set["combined_text"]

Y_train=strat_train_set["labels"]
Y_test=strat_test_set["labels"]

# checking if nan data is present
# print(X_train.isnull().sum())
# print(X_test.isnull().sum())

# Filling  the Nan values of dataset
X_train=X_train.fillna("").astype(str)
X_test=X_test.fillna("").astype(str)

 # converting my text and title to numbers so that my model could understand
vectorizer= TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)


 # training my model 
model = LogisticRegression()
model.fit(X_train_vec,Y_train)

# Predicting and evaluating the model
Y_pred= model.predict(X_test_vec)
# print("Accuracy of the model:",accuracy_score(Y_test,Y_pred))
# print("Classification Report:\n",classification_report(Y_test,Y_pred))
    
 
#  testing Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, Y_train)

Y_pred_nb = nb_model.predict(X_test_vec)

# print("🔹 Naive Bayes")
# print("Accuracy:", accuracy_score(Y_test, Y_pred_nb))
# print("Classification Report:\n", classification_report(Y_test, Y_pred_nb))


#  Model training by RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, Y_train)

Y_pred_rf = rf_model.predict(X_test_vec)

# print("🔹 Random Forest")
# print("Accuracy:", accuracy_score(Y_test, Y_pred_rf))
# print("Classification Report:\n", classification_report(Y_test, Y_pred_rf))


X = X.fillna('')
X = X.astype(str)
# Cross validating my model
# Logistic Regression Pipeline
logreg_pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_df=0.7),
    LogisticRegression()
)

# Naive Bayes Pipeline
nb_pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_df=0.7),
    MultinomialNB()
)

# Random Forest Pipeline
rf_pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_df=0.7),
    RandomForestClassifier(n_estimators=100, n_jobs=-1)
)

# Cross-validation
print("Performing cross-validation...")

logreg_scores = cross_val_score(logreg_pipeline, X, Y, cv=5, scoring='accuracy')
print("Logistic Regression Accuracy (CV):", logreg_scores.mean())

nb_scores = cross_val_score(nb_pipeline, X, Y, cv=5, scoring='accuracy')
print("Naive Bayes Accuracy (CV):", nb_scores.mean())

rf_scores = cross_val_score(rf_pipeline, X, Y, cv=5, scoring='accuracy')
print("Random Forest Accuracy (CV):", rf_scores.mean())


