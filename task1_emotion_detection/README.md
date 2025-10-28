````markdown
# Task 1 – Emotion Detection Using BERT  

this is my task 1 from **NLP / Machine Learning Project 2**.  
the goal was to train a BERT model that can detect emotions (joy, sadness, anger, neutral) from text.

---

## project overview  

in this task i fine-tuned a pre-trained **bert-base-uncased** model from hugging face on a labeled dataset of four emotions.  
after training, the model can predict the emotion behind any sentence or tweet-like text.

---

## 📂 dataset  

**source:** [Kaggle – Emotion Categories (Neutral, Joy, Sadness, Anger)](https://www.kaggle.com/datasets/faiqahmad01/emotion-categories-neutraljoysadnessanger)  
**total samples:** 22 050  

each record has:
- `content` → the text sentence  
- `sentiment` → numeric emotion label  

| label | emotion  |
|--------|-----------|
| 0 | Joy |
| 1 | Sadness |
| 2 | Neutral |
| 3 | Anger |

split ratio:
- 80 % train  
- 10 % validation  
- 10 % test  

---

##  setup  

run this project on **Google Colab** or any python 3.10+ environment.  
install all needed packages first:

```bash
pip install transformers datasets torch scikit-learn pandas matplotlib seaborn gradio
````

mount your drive and update the dataset paths in the notebook.

---

##  model details

* **base model:** bert-base-uncased
* **epochs:** 3
* **batch size:** 16 (train) / 32 (eval)
* **learning rate:** 2e-5
* **optimizer:** AdamW
* **metrics:** accuracy + F1 (macro & weighted)

training handled using `Trainer` and `TrainingArguments` from 🤗 Transformers.

---

## results

| metric        | score  |
| ------------- | ------ |
| accuracy      | ~ 0.91 |
| f1 (macro)    | ~ 0.90 |
| f1 (weighted) | ~ 0.91 |

✅ best model saved in `my_bert_emotion_model/`
✅ confusion matrix saved as `confusion_matrix.png`
✅ classification report saved as `classification_report.txt`
✅ summary saved as `results_summary.json`

---

##  example predictions

| text                                | predicted emotion |
| ----------------------------------- | ----------------- |
| “i’m so happy and excited today!”   | joy               |
| “this is the worst day of my life.” | sadness           |
| “the weather is nice today.”        | neutral           |
| “i’m furious about what happened!”  | anger             |

---

##  gradio demo

i built a simple interactive demo with **gradio** so anyone can test the model:

```python
import gradio as gr

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    return list(emotions.values())[pred]

gr.Interface(
    fn=predict_emotion,
    inputs="text",
    outputs="text",
    title="🎭 Emotion Detector",
    description="Detect Joy, Sadness, Anger, or Neutral from text. Made by Rameesha ❤️"
).launch()
```

this launches a small ui in colab with live predictions.

---

## what i learned

* how to fine-tune bert for classification
* how to use hugging face `Trainer`
* how to evaluate models with accuracy and F1
* how to build a quick gradio demo

---

## author

**Rameesha**
student | project 2 – Task 1 (Emotion Detection using BERT)
```

