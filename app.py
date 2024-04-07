import tkinter as tk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()
# transform the text
def transformText(text):
    text = text.lower()
    text = word_tokenize(text)

    words = []
    for word in text:
        if word.isalnum():   # return true if all the characters are alphanumeric
            words.append(word)

    text = words[:]
    words.clear()

    for word in text:
        if word not in stopwords.words('english'):
            words.append(word)

    text = words[:]
    words.clear()

    for word in text:
        words.append(ps.stem(word))

    

    return " ".join(words)

# tfdif = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))


df= pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
X = df['v2'].apply(transformText)
y = df['label']

tv = TfidfVectorizer(max_features=3000)
X = tv.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB() #NAIVE BAYES
mnb.fit(X_train,y_train)
mnb.score(X_test,y_test)


def hello():
    msg = msgval.get("1.0","end-1c")
    # transform_msg = transformText(msg) 
    transform_msg = transformText(msg) 

    vector_input = tv.transform([transform_msg]).toarray()
    print(vector_input)

    result = mnb.predict(vector_input)
    print(result[0])


    if result[0] == 1:
        lb1.config(text="Spam",fg="red")
    else:
        lb1.config(text="Not Spam",fg="green")
    
root = tk.Tk()

root.geometry("800x500")
root.title("SPAM CLASSIFIER test")
lb = tk.Label(text="Enter Text",font="comicsansms 20 bold")
lb.pack()

# msg = tk.StringVar()
msgval = tk.Text(root,height=20,width=90)
msgval.pack()

tk.Button(text="Predict",command=hello).pack(pady=5)


lb1 = tk.Label(text="",font="comicsansms 20 bold")
lb1.pack(pady=10)


root.mainloop()