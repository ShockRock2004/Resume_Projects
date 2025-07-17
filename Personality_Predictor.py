import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('mbti_1.csv')  # Kaggle MBTI Dataset


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text


df['clean_posts'] = df['posts'].apply(clean_text)
X = df['clean_posts']
y = df['type']  # MBTI Type (16 Classes)


vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_mbti(text):
    text_clean = clean_text(text)
    vect_text = vectorizer.transform([text_clean])
    prediction = model.predict(vect_text)[0]
    return prediction

# Example Usage
sample_text = "I love trying out new ideas and discussing theories with friends."
print("\nPredicted MBTI Type for sample input:", predict_mbti(sample_text))
