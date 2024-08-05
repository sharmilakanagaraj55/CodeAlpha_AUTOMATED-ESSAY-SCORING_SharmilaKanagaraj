import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

# Load the CSV file from the correct path with a different encoding
data = pd.read_csv('/content/essays.csv', encoding='latin1')

# Display the first few rows and the columns to understand the structure
print(data.head())
print(data.columns)

# Preprocess essays
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def preprocess_essay(essay):
    tokens = word_tokenize(essay)
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return ' '.join(filtered_tokens)

data['processed_essay'] = data['TEXT'].apply(preprocess_essay)  # Use 'TEXT' column
# Assuming the score is in 'score' column, if not, update accordingly
# For demonstration, let's assume we have a 'score' column
# If there's no 'score' column, you'll need to create or use another column
# y = data['score']  # Uncomment and modify this line if you have a score column

# For now, we'll generate some random scores for demonstration purposes
np.random.seed(42)
data['score'] = np.random.randint(1, 6, data.shape[0])  # Random scores between 1 and 5

X = data['processed_essay']
y = data['score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

# Load the CSV file from the correct path with a different encoding
data = pd.read_csv('/content/essays.csv', encoding='latin1')

# Display the first few rows and the columns to understand the structure
print(data.head())
print(data.columns)

# Preprocess essays
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def preprocess_essay(essay):
    tokens = word_tokenize(essay)
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return ' '.join(filtered_tokens)

data['processed_essay'] = data['TEXT'].apply(preprocess_essay)  # Use 'TEXT' column
# Assuming the score is in 'score' column, if not, update accordingly
# For demonstration, let's assume we have a 'score' column
# If there's no 'score' column, you'll need to create or use another column
# y = data['score']  # Uncomment and modify this line if you have a score column

# For now, we'll generate some random scores for demonstration purposes
np.random.seed(42)
data['score'] = np.random.randint(1, 6, data.shape[0])  # Random scores between 1 and 5

X = data['processed_essay']
y = data['score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Pipeline for preprocessing and model
model = Pipeline([
    ('tfidf', tfidf),
    ('regressor', Ridge())
])
# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
!pip install textstat
!pip install gramformer
!pip install transformers
!pip install tokenizers
print(data.head())
print(data.columns)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Pipeline for preprocessing and model
model = Pipeline([
    ('tfidf', tfidf),
    ('regressor', Ridge())
])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
