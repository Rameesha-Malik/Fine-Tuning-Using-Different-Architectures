# GPT-2 Recipe Generator

**Student:** Rameesha | **Course:** Advanced NLP (Project 02)

---

###  Overview

This project fine-tunes **GPT-2** to generate cooking recipes from a given title.
Type *“Chocolate Cake”* and the model writes ingredients and directions automatically.

---

### Dataset

* Source: Kaggle recipe dataset
* Fields: `title`, `ingredients`, `directions`
* Used ≈ 3 000 samples for quick training on Colab

---

### ⚙️ Model

| Item       | Value                  |
| ---------- | ---------------------- |
| Base Model | GPT-2 (124 M params)   |
| Optimizer  | AdamW                  |
| Epochs     | 2                      |
| LR         | 5e-5                   |
| Frameworks | PyTorch + Transformers |

---

### Results

| Metric        | Score |
| ------------- | ----- |
| BLEU          | 0.066 |
| ROUGE-1       | 0.248 |
| ROUGE-2       | 0.122 |
| Training Loss | 1.28  |

---

### 💻Quick Demo

```python
import gradio as gr

def generate_recipe(title):
    prompt = f"Title: {title} |"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_recipe,
    inputs="text",
    outputs="text",
    title="GPT-2 Recipe Generator by Rameesha"
)
demo.launch()
```

---

###  Example

**Input:** `Chocolate Cake`
**Output:**

```
Ingredients: flour, cocoa, eggs, sugar, butter
Directions: mix all and bake at 350 °F for 30 min
```

---

###  How to Run

1. Open Google Colab
2. Upload notebook + dataset to Drive
3. Run cells to train and save model
4. Launch Gradio demo

---

###  Tech Used

Python | PyTorch | Transformers | Pandas | TQDM | NLTK | ROUGE | Gradio


