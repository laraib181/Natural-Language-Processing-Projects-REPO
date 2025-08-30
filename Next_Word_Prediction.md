
# 🔮 Next Word Prediction  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)  
[![NLP](https://img.shields.io/badge/NLP-LSTM-green.svg)](#)  
[![Gradio](https://img.shields.io/badge/Gradio-UI%20Demo-lightgrey.svg)](https://gradio.app/)  

---

## 🔎 Overview  
**Next Word Prediction** is a Natural Language Processing (NLP) project that uses a **Long Short-Term Memory (LSTM)** model to predict the most probable next word in a given sequence of text.  

The model is trained on a custom text dataset to capture **linguistic patterns** and generate **context-aware predictions**.  

Applications include:  
- Text auto-completion (search engines, IDEs, mobile keyboards)  
- Chatbots & virtual assistants  
- Predictive typing & smart composition tools  

---

## ✨ Features  
✔️ Predicts the **next word** in a sentence  
✔️ Learns **long-term dependencies** with LSTM  
✔️ Provides **context-aware predictions**  
✔️ Modular design for **training & inference**  
✔️ Interactive **Gradio-based web interface**  

---

## 🛠️ Tools & Libraries  
- **Python 3.9+**  
- **TensorFlow / Keras** – Model building and training  
- **NumPy & Pandas** – Data preprocessing  
- **NLTK / Keras Tokenizer** – Tokenization and text handling  
- **Matplotlib / Seaborn** – Visualizations  
- **Gradio** – Model deployment interface  

---

## ⚙️ Techniques & Methodologies  
- **NLP Preprocessing:** Tokenization, text cleaning, sequence padding  
- **Deep Learning (LSTM):** Captures sequential dependencies  
- **Model Training Pipeline:**  
  1. Data preprocessing  
  2. Sequence generation  
  3. Training with categorical crossentropy  
  4. Evaluation & prediction  

---

## 🧠 Model Details  
The model is built using **Keras Sequential API**:  

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- **Embedding Layer** – Transforms words into dense vectors  
- **LSTM Layers** – Captures sequence dependencies  
- **Dropout** – Prevents overfitting  
- **Dense (Softmax)** – Outputs word probability distribution  

✅ Training Accuracy: **~95%** on structured dataset  

---

## 📂 Dataset  
- Custom **synthetic dataset** with structured sentences containing names, numbers, dates, and domain-specific text.  
- Preprocessed via **tokenization & padding**.  

Example entries:  
```
Dr. Smith observed 12.5 kg of data with 95% accuracy.
A student generated 20 samples on Dec. 3rd, 2021.
NASA calculated case no. 245B approx. 5 minutes later.
```

---

## 🚀 Installation & Usage  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/your-username/next-word-prediction.git
cd next-word-prediction
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Training Notebook  
```bash
jupyter notebook Next_Word_Prediction.ipynb
```

### 4️⃣ Example Model Usage  
```python
from keras.preprocessing.sequence import pad_sequences
import numpy as np

text = "The manager calculated"
token_text = tokenizer.texts_to_sequences([text])[0]
padded_text = pad_sequences([token_text], maxlen=56, padding='pre')
predicted_idx = np.argmax(model.predict(padded_text))

for word, idx in tokenizer.word_index.items():
    if idx == predicted_idx:
        print("Next word prediction:", word)
```

---

## 🌐 Gradio Interface  

This project includes a **Gradio interface** for interactive predictions.  

```python
import gradio as gr

def predict_next_word(text):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_text = pad_sequences([token_text], maxlen=56, padding='pre')
    pred_idx = np.argmax(model.predict(padded_text))
    for word, idx in tokenizer.word_index.items():
        if idx == pred_idx:
            return word

demo = gr.Interface(
    fn=predict_next_word,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=gr.Textbox(label="Predicted Next Word"),
    title="Next Word Prediction with LSTM",
    description="Type a sentence and get the predicted next word."
)

demo.launch()
```

👉 Running this will launch a **web app** where you can type a sentence and instantly see predictions.  

---

## 📊 Results  
- **Accuracy:** ~95% on structured test data  
- **Strength:** Predicts contextually appropriate next words  
- **Visualization:** Training/validation curves show stable convergence  

---

## 📝 Conclusion  
The **Next Word Prediction** project demonstrates how **LSTM models** can effectively model sequential dependencies in text.  
It validates the power of deep learning in **predictive NLP tasks** and sets the foundation for real-world applications.  

---

## 🔮 Future Improvements & Limitations  
- Train on **larger natural text corpora** for better generalization  
- Add **beam search & top-k sampling** for richer predictions  
- Benchmark against **Transformer models (BERT, GPT, etc.)**  
- Deploy as a **web API / mobile app** for real-time usage  

---

📌 *This repository is for research and educational purposes in Natural Language Processing (NLP).*  
