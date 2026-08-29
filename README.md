# AG News Classification & Headline Generation

An NLP project using the **AG News dataset** to explore news categorization and automated headline generation with **PEGASUS**, a transformer-based sequence-to-sequence model.

## 📌 Project Overview

This project focuses on generating concise headlines from news article text using a fine-tuned **Google PEGASUS (`google/pegasus-cnn_dailymail`)** model.

The AG News dataset contains news articles from four categories:

* 🌎 World
* ⚽ Sports
* 💼 Business
* 🔬 Sci/Tech

Because the AG News dataset does not provide separate headline labels, the project creates **pseudo-headlines** by extracting and cleaning the first sentence of each article. The remaining article content is used as the description/input for headline generation.

## 🎯 Objectives

* Preprocess and clean AG News articles.
* Create pseudo-headline and article-description pairs.
* Fine-tune PEGASUS for headline generation.
* Evaluate generated headlines using **ROUGE** and **BERTScore**.
* Analyze performance across different news categories.
* Compare multiple decoding strategies for headline generation.

## 🧠 Model

The project uses:

**Model:** `google/pegasus-cnn_dailymail`

PEGASUS is a transformer-based sequence-to-sequence model designed for abstractive text summarization. It is fine-tuned in this project to generate concise news headlines from article descriptions.

### Training Configuration

* Learning rate: `3e-5`
* Epochs: `6`
* Training batch size: `16`
* Evaluation batch size: `16`
* Maximum input length: `128`
* Maximum output length: `64`
* Weight decay: `0.01`
* Warmup steps: `500`
* Beam search: `6 beams`
* Length penalty: `2.0`
* No-repeat n-gram size: `3`
* Mixed precision (FP16): enabled when CUDA is available

## 🔄 Data Processing

The original AG News article text is transformed into training pairs:

```text
Article
   ↓
Sentence extraction
   ↓
First sentence → Pseudo-headline
Remaining sentences → Description
   ↓
PEGASUS fine-tuning
   ↓
Generated headline
```

The dataset is divided into:

* **80% Training**
* **20% Validation**
* **Official AG News Test Set**

A random seed of `42` is used for reproducibility.

## 📊 Evaluation

The generated headlines are evaluated using:

### ROUGE

* ROUGE-1
* ROUGE-2
* ROUGE-L

### BERTScore

BERTScore is used to measure semantic similarity between generated headlines and reference pseudo-headlines.

Evaluation is performed on both validation and test data, including category-wise analysis for:

* World
* Sports
* Business
* Sci/Tech

## 🔍 Decoding Strategy Comparison

The project also compares several generation strategies:

* Greedy decoding
* Temperature sampling (`0.77`)
* Top-k sampling (`k=50`)
* Top-p sampling (`p=0.9`)
* Beam search

The strategies are evaluated using ROUGE-L and BERTScore.

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* PEGASUS
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn

## 📁 Project Structure

```text
ag-news-classification-headline-generation/
│
├── ag_news_classification_headline_generation.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Shreyashree-Mondal/ag-news-classification-headline-generation.git
cd ag-news-classification-headline-generation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python ag_news_classification_headline_generation.py
```

The script downloads the AG News dataset and required NLTK resources, loads the PEGASUS model, performs preprocessing, fine-tunes the model, evaluates performance, and generates evaluation visualizations.

> **Note:** Fine-tuning PEGASUS on the full dataset is computationally intensive and is best performed with a CUDA-enabled GPU and sufficient GPU memory.

## 📈 Outputs

The project produces:

* Training and validation loss curves
* Epoch-aligned loss comparison
* Generated headline examples
* Overall ROUGE scores
* Overall BERTScore
* Category-wise evaluation
* Validation decoding-strategy comparisons
* Test decoding-strategy comparisons

## 👩‍💻 Author

**Shreyashree Mondal**

Master's in Data Science and Artificial Intelligence
University of Houston

[GitHub](https://github.com/Shreyashree-Mondal)
