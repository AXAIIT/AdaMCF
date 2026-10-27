---
license: apache-2.0
library_name: keras
---
# Mobilenet-v2-CIFAR-10 (Teeny-Tiny Castle)

This model is part of a tutorial tied to the [Teeny-Tiny Castle](https://github.com/Nkluge-correa/TeenyTinyCastle), an open-source repository containing educational tools for AI Ethics and Safety research. 

## How to Use

```python
from transformers import pipeline
from datasets import load_dataset
from IPython.display import display

dataset = load_dataset("cifar10", split="test")

classifier = pipeline("image-classification", model="AiresPucrs/Mobilenet-v2-CIFAR-10")

display(dataset['img'][5].resize((256, 256)))
display(classifier(dataset['img'][5]))

display(dataset['img'][10].resize((256, 256)))
display(classifier(dataset['img'][10]))
```