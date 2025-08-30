# 📰 Named Entity Recognition (NER) from News Articles

## 📌 Overview

This project focuses on **extracting named entities** (e.g., people,
locations, organizations, and miscellaneous entities) from **news
articles** using both **rule-based** and **deep learning approaches**.\
The goal is to highlight and classify entities in text for better
information retrieval, text analytics, and downstream NLP applications.

------------------------------------------------------------------------

## ✨ Features

-   Support for **rule-based** NER using spaCy custom pipelines.\
-   **Transformer-based model (BERT)** for high-accuracy NER.\
-   Comparison between **multiple approaches**: rule-based, spaCy models
    (small & large), and BERT.\
-   **Evaluation metrics**: Precision, Recall, and F1-Score (via
    `seqeval`).\
-   **Gradio interface** for interactive model comparison.\
-   Pretrained **CoNLL-2003 dataset** integration for training and
    testing.

------------------------------------------------------------------------

## 🛠️ Tools & Libraries

-   **Python 3.12+**\
-   [PyTorch](https://pytorch.org/)\
-   [Hugging Face Transformers](https://huggingface.co/transformers/)\
-   [spaCy](https://spacy.io/)\
-   [seqeval](https://github.com/chakki-works/seqeval)\
-   \[pandas, numpy, scikit-learn\]\
-   [Gradio](https://www.gradio.app/)

------------------------------------------------------------------------

## 🔬 Techniques & Methodologies

-   **Token Classification** using **BERT-base-cased** with
    fine-tuning.\
-   **Rule-based heuristics** for entity recognition.\
-   **spaCy small (en_core_web_sm) & large (en_core_web_lg) models**.\
-   Training with **AdamW optimizer** and linear learning rate
    scheduling.\
-   Evaluation on standard **CoNLL-2003 benchmark dataset**.\
-   Visualization of entities using **spaCy displacy**.

------------------------------------------------------------------------

## 📊 Dataset

-   **CoNLL-2003 Dataset** (commonly used for NER).
    -   Entities: `PER, ORG, LOC, MISC`.\
    -   Provided in train/test/validation splits.\
-   Example Sentence:\
    \> *"Barack Obama met Angela Merkel in Berlin"* → Entities: `PER`,
    `PER`, `LOC`

------------------------------------------------------------------------

## ⚙️ Installation & Usage

### 1. Clone Repository

``` bash
git clone https://github.com/yourusername/NER-NewsArticles.git
cd NER-NewsArticles
```

### 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3. Download Dataset

-   Place CoNLL-2003 dataset files (`train.txt`, `valid.txt`,
    `test.txt`) in the `data/` folder.

### 4. Train Model

``` bash
python train.py
```

### 5. Run Gradio Interface

``` bash
python app.py
```

Open the local URL to interact with the models.

------------------------------------------------------------------------

## 📈 Results

-   **BERT model achieved an F1-score of \~94.6%** on validation data.\
-   **spaCy large model** outperformed rule-based and small models.\
-   Rule-based NER worked well for simple entities but lacked
    generalization.

  Model         Precision   Recall     F1-Score
  ------------- ----------- ---------- ----------
  Rule-Based    \~0.70      \~0.65     \~0.67
  spaCy Small   \~0.85      \~0.84     \~0.84
  spaCy Large   \~0.89      \~0.88     \~0.88
  **BERT**      **0.95**    **0.94**   **0.95**

------------------------------------------------------------------------

## ✅ Conclusion

This project demonstrates the **strength of transformer-based models**
for Named Entity Recognition compared to rule-based or classical NLP
methods. The interactive **Gradio app** enables practical exploration
and comparison of multiple approaches.

------------------------------------------------------------------------

## 🚀 Future Improvements & Limitations

### Limitations:

-   High computation requirements for transformer models.\
-   Limited to **CoNLL-2003 entities**; may miss domain-specific
    entities.

### Future Work:

-   Extend to **multi-lingual NER**.\
-   Fine-tune with **domain-specific datasets** (e.g., finance,
    healthcare).\
-   Deploy as an **API for real-time entity extraction**.\
-   Incorporate **NER + Relation Extraction** for deeper text
    understanding.

------------------------------------------------------------------------

## 📚 References

-   CoNLL-2003 Shared Task: [Dataset
    Info](https://www.clips.uantwerpen.be/conll2003/ner/)\
-   Hugging Face Transformers Documentation\
-   spaCy Official Docs

------------------------------------------------------------------------

## 🙌 Acknowledgments

Special thanks to open-source communities of **Hugging Face**,
**spaCy**, and **PyTorch** for providing state-of-the-art NLP tools.
