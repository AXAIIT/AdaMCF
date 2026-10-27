---
language: en
license: mit
library_name: timm
tags:
- image-classification
- resnet18
- cifar10
datasets: cifar10
metrics:
- accuracy
model-index:
- name: resnet18_cifar10
  results:
  - task:
      type: image-classification
    dataset:
      name: CIFAR-10
      type: cifar10
    metrics:
    - type: accuracy
      value: 0.9498
---

# Model Card for Model ID

This model is a small resnet18 trained on cifar10.

- **Test Accuracy:** 0.9498
- **License:** MIT

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import detectors
import timm

model = timm.create_model("resnet18_cifar10", pretrained=True)
```

## Training Data

Training data is cifar10.

## Training Hyperparameters


- **config**: `scripts/train_configs/cifar10.json`

- **model**: `resnet18_cifar10`

- **dataset**: `cifar10`

- **batch_size**: `128`

- **epochs**: `300`

- **validation_frequency**: `5`

- **seed**: `1`

- **criterion**: `CrossEntropyLoss`

- **criterion_kwargs**: `{}`

- **optimizer**: `SGD`

- **lr**: `0.1`

- **optimizer_kwargs**: `{'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': 'True'}`

- **scheduler**: `ReduceLROnPlateau`

- **scheduler_kwargs**: `{'factor': 0.1, 'patience': 3, 'threshold': 0.001, 'mode': 'max'}`

- **debug**: `False`


## Testing Data

Testing data is cifar10.

---

This model card was created by Eduardo Dadalto.