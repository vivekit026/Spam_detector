**📌 Project Title**
AI-Based Spam Message Detection System Using NLP and Naive Bayes

**🎯 Problem Statement**
Spam messages and emails waste time and may cause fraud. This project aims to automatically classify messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and Naive Bayes algorithm.

**🧠 Objective**
To analyze text messages

To extract useful features using NLP

To classify messages using a machine learning model

**🛠️ Technologies Used**
Programming Language: Python

Libraries:
numpy, pandas
nltk
scikit-learn
Algorithm: Multinomial Naive Bayes
Dataset: SMS Spam Collection Dataset

**📂 Dataset Description**
Total messages: ~5,500
Labels:
spam
ham (not spam)
Example:

makefile
Copy code
spam: Win a free mobile now!
ham: Are we meeting today?

**🔄 System Workflow**
Input SMS / Email text
Text Preprocessing
Feature Extraction (Bag of Words / TF-IDF)
Model Training (Naive Bayes)
Spam or Ham Prediction

**⚙️ Implementation Steps**
1️⃣ Import Required Libraries
python
Copy code
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
2️⃣ Load Dataset
python
Copy code
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
3️⃣ Text Preprocessing
python
Copy code
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['message'] = data['message'].apply(preprocess)
4️⃣ Feature Extraction
python
Copy code
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']
5️⃣ Train-Test Split
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
6️⃣ Train Naive Bayes Model
python
Copy code
model = MultinomialNB()
model.fit(X_train, y_train)
7️⃣ Model Evaluation
python
Copy code
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
✔️ Expected Accuracy: 95%+

8️⃣ Test with Custom Message
python
Copy code
msg = ["Congratulations! You won a free lottery"]
msg_vector = vectorizer.transform(msg)
print(model.predict(msg_vector))

**📊 Output**
Spam → Promotional / fraudulent message
Ham → Genuine message
🧪 Algorithm Used: Naive Bayes
Why Naive Bayes?
Fast and efficient
Works well for text classification
Requires less training data
 
**✅ Advantages**
High accuracy
Low computational cost
Easy to implement

**❌ Limitations**
Assumes word independence
Less effective for very short messages

**🚀 Future Enhancements**
Use Deep Learning (LSTM / BERT)
Add email attachment scanning
Multilingual spam detection
Deploy as a web or mobile app
