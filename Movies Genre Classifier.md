# Movies Genre Classifier

### Overview
The **Movies Genre Classifier** is an NLP-based machine learning project designed to categorize movies into genres using textual descriptions. It also includes a Gradio-powered interface to make real-time predictions directly from user input.

### Features
- Cleans and processes raw text data.
- Applies Bag of Words and TF-IDF to extract features from text.
- Trains a Multinomial Naive Bayes classifier for genre prediction.
- Provides a Gradio interface for live predictions.
- Evaluates model performance with visual results.

### Tools & Libraries
- **Python** – Programming language for the project.
- **Pandas & NumPy** – Data manipulation and numerical processing.
- **Scikit-learn** – Machine learning and vectorization techniques.
- **Matplotlib & Seaborn** – Visual representation of data and results.
- **NLTK** – Natural Language Processing toolkit for text cleaning and tokenization.
- **Gradio** – Interactive interface for model predictions.

### Techniques & Methodologies
1. **Data Preprocessing**
   - Remove special characters, punctuations, and symbols.
   - Convert text to lowercase for uniformity.
   - Tokenize text and remove stopwords to retain meaningful words.
   - Lemmatize words to their root form for better feature extraction.

2. **Feature Extraction**
   - Bag of Words (BoW): Represents text as word frequency.
   - TF-IDF (Term Frequency-Inverse Document Frequency): Highlights important words by reducing the weight of commonly used ones.

3. **Modeling**
   - Multinomial Naive Bayes (MultinomialNB) is used for text classification due to its simplicity and efficiency with high-dimensional data.

4. **Gradio Interface**
   - Provides a lightweight web app where users can input a movie description and get predicted genres instantly.

### Dataset
- Dataset includes movie descriptions and scripts.
- Cleaned and processed to remove noise and improve accuracy.
- Transformed into numerical vectors for model compatibility.

### Installation & Usage
```bash
git clone <repository_url>
cd movies-genre-classifier
pip install -r requirements.txt
python main.py
```
To launch Gradio interface:
```bash
python app.py
```

### Results
- Visual insights include genre distribution plots and classification accuracy.
- Provides a baseline accuracy for future improvements.

### Conclusion
The project successfully demonstrates how NLP combined with machine learning can classify movies into genres. The integration of Gradio adds interactivity, making it easier to test and deploy the model in real-world scenarios.

### Future Improvements & Limitations
- **Limitations**:
  - Model performance may degrade with highly imbalanced datasets.
  - Naive Bayes is limited in capturing context and semantic meaning.
- **Future Improvements**:
  - Incorporate deep learning methods (LSTM, GRU, BERT).
  - Utilize word embeddings for richer semantic representation.
  - Experiment with ensemble techniques for improved accuracy.
  - Expand the dataset to include multi-genre classification scenarios.
  - Enhance Gradio interface with better UI/UX design and cloud deployment options.
