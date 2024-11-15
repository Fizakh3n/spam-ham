# Spam-Ham Text Classification with Flask


This project implements a **Spam vs Ham** text classification model using a **Naive Bayes classifier**. The model is built using **scikit-learn** and is deployed as a web application with **Flask**. The web app allows users to input text (such as an email or message) and classify it as either **Spam** or **Ham**. The classification is done in real-time, and the result is displayed on the web interface.
---
## Features
- Text preprocessing (lowercasing, removing punctuation and numbers, and stopwords removal)
- CountVectorizer for converting text into numerical features
- Naive Bayes classifier (MultinomialNB) for spam-ham classification
- Web interface built using Flask
- Real-time classification feedback
  ---
## Technologies Used
- **Python 3.x**
- **Flask**: Web framework for serving the app
- **scikit-learn**: Machine learning library for model and vectorization
- **NLTK**: Natural Language Toolkit for text processing (stopwords removal)
- **Pickle**: For saving and loading the trained model
  ---
  ## Setup Instructions

### Prerequisites
1. **Python 3.x** (Use version 3.8 or higher)
2. **Install dependencies**:
   - First, clone the repository:
     ```bash
     git clone https://github.com/yourusername/spam-ham-classification.git
     cd spam-ham-classification
     ```

   - Then, install the required libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download NLTK stopwords** by running the following Python code:
   ```python
   import nltk
   nltk.download('stopwords')
