# GPT-2 Recipe Generator
---

###  Overview

This project fine-tunes **GPT-2** to generate cooking recipes from a given title.
Type *“Chocolate Cake”* and the model writes ingredients and directions automatically.

---

### Dataset

* Source: Kaggle recipe dataset [https://www.kaggle.com/datasets/nazmussakibrupol/3a2mext/data]
* Fields: `title`, `ingredients`, `directions`
* Used ≈ 3 000 samples for quick training on Colab

---

###  Model

| Item       | Value                  |
| ---------- | ---------------------- |
| Base Model | GPT-2 (124 M params)   |
| Optimizer  | AdamW                  |
| Epochs     | 2                      |
| LR         | 5e-5                   |
| Frameworks | PyTorch + Transformers |

---
### Saved finetuned model link

https://huggingface.co/rameesha146/gpt2_recipe_model

### Deploy link

https://huggingface.co/spaces/rameesha146/recipe-generation-finetune-gpt-2

---

###  Tech Used

Python | PyTorch | Transformers | Pandas | TQDM | NLTK | ROUGE | Gradio


