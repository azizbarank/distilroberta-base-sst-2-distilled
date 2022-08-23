## Installing the Necessary Packages


```python
%pip install "pytorch==1.10.1"
%pip install transformers datasets tensorboard --upgrade

!sudo apt-get install git-lfs
```


```python
from huggingface_hub import notebook_login # Logging into our Hugging Face account to store the acces token

notebook_login()
```

## Defining the Teacher and the Student


```python
student_id = "distilroberta-base"
teacher_id = "textattack/roberta-base-SST-2"

# name for our repository
repo_name = "distilroberta-base-sst2-distilled"
```


```python
# The code to ensure that the same output is created by the teacher and the student
from transformers import AutoTokenizer

# initializing tokenizer
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
student_tokenizer = AutoTokenizer.from_pretrained(student_id)

# sample input
sample = "This is a basic example, with different words to test."

# assert results
assert teacher_tokenizer(sample) == student_tokenizer(sample), "Tokenizers haven't created the same output"
```

## Loading Our Data


```python
dataset_id="glue"
dataset_config="sst2"
```


```python
# loading the dataset
from datasets import load_dataset

dataset = load_dataset(dataset_id,dataset_config)
```

## Tokenization


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(teacher_id)
```


```python
def process(examples):
    tokenized_inputs = tokenizer(
        examples["sentence"], truncation=True, max_length=512
    )
    return tokenized_inputs

tokenized_datasets = dataset.map(process, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label","labels")

tokenized_datasets["test"].features

```

## Distillation of the Model with PyTorch


```python
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

```

## Hyperparameters and Loading the Model


```python
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from huggingface_hub import HfFolder

# label2id, id2label dicts for the outputs for the model
labels = tokenized_datasets["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# training args
training_args = DistillationTrainingArguments(
    output_dir=repo_name,
    num_train_epochs=7,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    fp16=True,
    learning_rate=6e-5,
    seed=33,
    logging_dir=f"{repo_name}/logs",
    logging_strategy="epoch", 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    hub_token=HfFolder.get_token(),
    # distilation parameters
    alpha=0.5,
    temperature=4.0
    )

# data_collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# model
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# student model
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

```

## Metrics


```python
# installing sklearn for the metrics
!pip install sklearn 
```


```python
from datasets import load_metric
import numpy as np

# metrics and metrics function
accuracy_metric = load_metric( "accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
    }

```

## Training


```python
trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

```


```python
trainer.train()

```

## Hyperparameter Tuning using Optuna


```python
%pip install optuna

```


```python
def hp_space(trial):
    return {
      "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
      "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3 ,log=True),
      "alpha": trial.suggest_float("alpha", 0, 1),
      "temperature": trial.suggest_int("temperature", 2, 30),
      }

```


```python
def student_init():
    return AutoModelForSequenceClassification.from_pretrained(
        student_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

trainer = DistillationTrainer(
    model_init=student_init,
    args=training_args,
    teacher_model=teacher_model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
best_run = trainer.hyperparameter_search(
    n_trials=2,
    direction="maximize",
    hp_space=hp_space
)

print(best_run)
```

## Saving the best model and pushing it to the Hub


```python
# overwriting the initial hyperparameters
for k,v in best_run.hyperparameters.items():
    setattr(training_args, k, v)

# defining new repository to store our distilled model
best_model_ckpt = "distilroberta-best"
training_args.output_dir = best_model_ckpt
```


```python
# creating a new Trainer with optimal parameters
optimal_trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

optimal_trainer.train()


# saving best model and metrics 
trainer.create_model_card(model_name=training_args.hub_model_id)
trainer.push_to_hub()
```
