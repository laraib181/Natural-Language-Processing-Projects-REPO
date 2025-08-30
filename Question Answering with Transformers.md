# ❓ Question Answering with Transformers

## 📌 Overview

This project focuses on building a **Question Answering (QA) system**
using **Transformer-based architectures**, specifically **BERT
(Bidirectional Encoder Representations from Transformers)**.

The model takes a **context passage** and a **natural language
question** as input, and returns the **most relevant answer span** from
the passage.\
It demonstrates how pre-trained Transformer models can be fine-tuned for
**machine reading comprehension** tasks using benchmark datasets.

------------------------------------------------------------------------

## 🔎 What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is a
pre-trained deep learning model developed by Google.\
Unlike traditional models that read text **left-to-right** or
**right-to-left**, BERT uses a **bidirectional self-attention
mechanism** to understand the full context of each word in relation to
all others in the sequence.

-   **Architecture**:
    -   Based on the **Transformer encoder**.\
    -   Uses **multi-head self-attention** to capture semantic
        relationships.\
    -   Pre-trained with **Masked Language Modeling (MLM)** and **Next
        Sentence Prediction (NSP)** tasks.
-   **For Question Answering**:
    -   Given a passage + question, BERT outputs probability scores for
        the **start** and **end positions** of the answer within the
        passage.\
    -   The answer span is extracted by selecting tokens with the
        highest probability.

------------------------------------------------------------------------

## ✨ Features

-   Fine-tuning of **BERT-base-uncased** on **SQuAD v1.1** dataset.\
-   Contextual understanding with **bidirectional attention**.\
-   Extractive QA: predicts exact **answer spans** from passages.\
-   Supports **GPU acceleration** for faster training.\
-   Evaluation with **Exact Match (EM)** and **F1 Score**.\
-   Modular code structure for training, evaluation, and inference.

------------------------------------------------------------------------

## 🛠️ Tools & Libraries

-   **Python 3.12+**\
-   [PyTorch](https://pytorch.org/) -- deep learning framework\
-   [Hugging Face Transformers](https://huggingface.co/transformers/) --
    pre-trained BERT\
-   [Hugging Face Hub](https://huggingface.co/) -- model and tokenizer
    management\
-   [SQuAD v1.1 Dataset](https://rajpurkar.github.io/SQuAD-explorer/) --
    QA dataset\
-   \[numpy, pandas\] -- data handling and preprocessing

------------------------------------------------------------------------

## 🔬 Techniques & Methodologies

-   **Tokenization** with BERT tokenizer for context + question pairs.\
-   **Fine-tuning strategy**:
    -   Optimizer: **AdamW**\
    -   Learning rate: `5e-5`\
    -   Batch size: 16\
    -   Epochs: 3\
-   **Answer span extraction**: model predicts `start_position` and
    `end_position` in the passage.\
-   **Loss function**: Cross-entropy over start and end token
    positions.\
-   **Evaluation** with EM and F1 for accuracy measurement.

------------------------------------------------------------------------

## 📊 Dataset

-   **Stanford Question Answering Dataset (SQuAD v1.1)**

    -   Train set: **87,599 question-answer pairs**.\
    -   Validation set: **34,726 question-answer pairs**.\

-   Entities are manually annotated with precise answer spans.\

-   Example:

    **Context:**\
    \> "Architecturally, the school has a Catholic character..."

    **Question:**\
    \> "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes
    France?"

    **Answer:**\
    \> *"Saint Bernadette Soubirous"*

------------------------------------------------------------------------

## ⚙️ Installation & Usage

### 1. Clone Repository

``` bash
git clone https://github.com/yourusername/QA-Transformers.git
cd QA-Transformers
```

### 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3. Download Dataset

``` bash
mkdir squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad/dev-v1.1.json
```

### 4. Train the Model

``` bash
python train.py
```

### 5. Run Inference

``` python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("./saved_model")

context = "Barack Obama served as the 44th President of the United States."
question = "Who served as the 44th President of the United States?"

inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
with torch.no_grad():
    start_scores, end_scores = model(**inputs).to_tuple()

all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
answer = " ".join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])
print("Answer:", answer)
```

------------------------------------------------------------------------

## 📈 Results

-   **Training Loss** steadily decreased over epochs.\
-   Validation results indicate strong generalization:

  Metric            Score
  ----------------- -----------
  **F1 Score**      \~85--90%
  **Exact Match**   \~80--85%

-   Qualitative results show the model correctly identifies **precise
    answer spans** in diverse contexts.

------------------------------------------------------------------------

## ✅ Conclusion

This project demonstrates how **BERT's bidirectional contextual
understanding** enables accurate **extractive QA**.\
Fine-tuning on **SQuAD v1.1** allows the model to achieve **high EM and
F1 scores**, proving Transformers' effectiveness for **machine reading
comprehension** tasks.

------------------------------------------------------------------------

## 🚀 Future Improvements & Limitations

### Limitations:

-   High memory and computation cost for large-scale training.\
-   Performance may degrade for **out-of-domain questions**.

### Future Work:

-   Extend to **SQuAD v2.0** (handling unanswerable questions).\
-   Experiment with **RoBERTa, ALBERT, DistilBERT** for efficiency
    vs. accuracy trade-offs.\
-   Develop an **interactive web API** for real-world applications.\
-   Explore **generative QA** with LLMs like GPT.

------------------------------------------------------------------------

## 📚 References

-   Devlin et al., 2018: [BERT: Pre-training of Deep Bidirectional
    Transformers for Language
    Understanding](https://arxiv.org/abs/1810.04805)\
-   Rajpurkar et al., 2016: [SQuAD: 100,000+ Questions for Machine
    Comprehension of Text](https://arxiv.org/abs/1606.05250)\
-   Hugging Face Transformers Documentation\
-   PyTorch Official Documentation

------------------------------------------------------------------------

## 🙌 Acknowledgments

Thanks to the **Hugging Face** and **PyTorch** communities for providing
state-of-the-art tools that made this project possible.
