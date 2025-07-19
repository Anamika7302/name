import os
import joblib

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report

MODEL_FILE = "model.pkl"
PIPELINE_FILE ="pipeline.pkl"

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("news1.csv", dtype=str)
    data = data.loc[:, [not str(col).startswith('Unnamed') for col in data.columns]]
     
    data["title"]=data['title'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['text'] = data['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data=data.drop("label",axis=1)
    
    data["combined_text"]=data["title"] +" " + data["text"]

    # Specifiying features and labels
    X =data["combined_text"]
    Y =data["labels"]
    X = X.fillna('')
    X = X.astype(str)

    split =StratifiedShuffleSplit(n_splits= 1, test_size=0.2,random_state=42)

    for train_index , test_index in split.split(data, data["labels"]):
     strat_train_set = data.loc[train_index]
     strat_test_set = data.loc[test_index]
    
#  redifining the  features and labels after splitting for training and testing
    X_train=strat_train_set["combined_text"]
    X_test=strat_test_set["combined_text"]

    Y_train=strat_train_set["labels"]
    Y_test=strat_test_set["labels"]


# Filling  the Nan values of dataset
    X_train=X_train.fillna("").astype(str)
    X_test=X_test.fillna("").astype(str)

 # converting my text and title to numbers so that my model could understand
    vectorizer= TfidfVectorizer(stop_words='english',max_df=0.7)
    X_train_vec=vectorizer.fit_transform(X_train)
    X_test_vec=vectorizer.transform(X_test)

    #  Model training by RandomForestClassifier
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', max_df=0.7),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )

    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)
    print("🔹 Random Forest")
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
     
    joblib.dump(pipeline.named_steps['randomforestclassifier'], MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE) 

    print("Model trained and saved.")


else:
    # INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv",dtype=str)
    input_data = input_data.loc[:, [not str(col).startswith('Unnamed') for col in input_data.columns]]
    
    input_data["title"] = input_data['title'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    input_data['text'] = input_data['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    input_data["combined_text"] = input_data["title"].fillna("") + " " + input_data["text"].fillna("")

    # Predict using pipeline
    predictions = pipeline.predict(input_data["combined_text"].astype(str))
    input_data["labels"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")