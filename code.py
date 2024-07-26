!pip install nltk 
import nltk 
#NATURAL LANGUAGE TOOK KIT LIBRARY for language processing tasks nltk.download('stopwords') #words like the is and which dont add nessesary meaning nltk.download('punkt') #breaking down text in words or texts 
#importing nessesary libraries 
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
# Load the fake and real news datasets 
fake_news = pd.read_csv("/content/Fake (2).csv") 
real_news = pd.read_csv("/content/True (2).csv") 
# Preprocess the text data 
def preprocess_text(text): 
text = text.lower() 
tokens = word_tokenize(text) #individual words
stop_words = set(stopwords.words("english")) #remove stop words tokens = [token for token in tokens if token not in stop_words] #filters stop words from tokens 
return " ".join(tokens) #returns as single text 
#preprocessing both datas 
fake_news["text"] = fake_news["text"].apply(preprocess_text) 
real_news["text"] = real_news["text"].apply(preprocess_text) 
# Create a TF-IDF vectorizer 
vectorizer = TfidfVectorizer() # TF-how frequently a term (word) occurs in a document. #IDF-how important a term is across all documents 
# Fit the vectorizer on the combined dataset 
X = vectorizer.fit_transform(list(fake_news["text"]) + list(real_news["text"])) 
# Split the data into training and testing sets 
y = np.concatenate((np.ones(len(fake_news)), np.zeros(len(real_news)))) X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
# Train a logistic regression model 
model = LogisticRegression() 
model.fit(X_train, y_train) 
# Evaluate the model on the testing set 
score = model.score(X_test, y_test) 
print("Accuracy:", score) 
# Predict the label for a new piece of text 
def predict_fake_news(text): 
text = preprocess_text(text) 
X_new = vectorizer.transform([text]) 
prediction = model.predict(X_new)[0] 
if prediction == 1: 
return "The above news is Fake News" 
else: 
return "The above news is Real News"
# Example usage 
text = "WASHINGTON - The U.S. Senate has passed a bipartisan infrastructure bill aimed at modernizing the national transportation system. The bill allocates funding for repairing roads, bridges, and public transit systems. Senate Majority Leader John Smith praised the bill as a historic investment in the nation's future, emphasizing job creation and economic growth. The bill includes provisions for expanding public transit, investing in renewable energy, and addressing environmental challenges. President Jane Doe intends to sign the bill into law, highlighting its importance for America's future. The bill now moves to the House for consideration." 
prediction = predict_fake_news(text) 
print(prediction) 
