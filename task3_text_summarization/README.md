# Text Summarization with T5 — Task 3

This project demonstrates how to fine-tune **T5 (Text-To-Text Transfer Transformer)** for **automatic text summarization** using the **CNN/Daily Mail dataset**. The model takes long news articles and generates concise summaries that capture the main ideas.

---

## What is Text Summarization?

**Text Summarization** is the process of shortening a long piece of text while preserving its key information and meaning. It is a common **Natural Language Processing (NLP)** task used in applications like news aggregation, report generation, and content curation.

---

##  Model Overview

* **Model Used:** `t5-base` (220M parameters)
* **Framework:** PyTorch + Hugging Face Transformers
* **Tokenizer:** T5 tokenizer
* **Task Type:** Abstractive Summarization
* **Dataset:** CNN/Daily Mail
* **Hardware:** Google Colab GPU (T4)
* **Training Time:** ≈ 45 minutes

---

## Dataset

The **CNN/Daily Mail dataset** consists of thousands of real news articles paired with human-written highlights (summaries).

Each sample includes:

* **article:** Full news story text
* **highlights:** Short human-written summary

For this project:

* **Training samples:** 10,000
* **Validation samples:** 1,000

---

## Training Configuration

| Parameter             | Value                     |
| --------------------- | ------------------------- |
| **Epochs**            | 3                         |
| **Batch Size**        | 8                         |
| **Learning Rate**     | 2e-5                      |
| **Max Input Length**  | 512                       |
| **Max Target Length** | 128                       |
| **Metric**            | ROUGE-1, ROUGE-2, ROUGE-L |

---

## 🧩 Code Workflow

### Load and Clean Data

```python
train_df = pd.read_csv('train.csv')
train_df = train_df.dropna(subset=['article', 'highlights'])
```

### Tokenization

```python
def prepare_data(examples):
    model_inputs = tokenizer(
        ["summarize: " + doc for doc in examples['article']],
        max_length=512, truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

### Trainer Setup

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./t5_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    predict_with_generate=True,
    evaluation_strategy="epoch"
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)
trainer.train()
```

### Evaluation

Model performance was evaluated using **ROUGE** metrics, which compare generated summaries with reference summaries.

---

## Learning Outcomes

* Gained hands-on experience fine-tuning **T5** for text generation
* Learned to use **Hugging Face Trainer** for Seq2Seq tasks
* Understood how **ROUGE** metrics evaluate summarization quality
* Built a deployable summarization model

---

## Conclusion

T5 effectively summarizes long texts into concise, meaningful statements.
With larger datasets and longer training, its performance can approach **human-level summarization quality**.

