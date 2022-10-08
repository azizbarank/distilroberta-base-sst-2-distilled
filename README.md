# distilroberta-base-sst-2-distilled

This is the distilled version of the [RoBERTa](https://huggingface.co/textattack/roberta-base-SST-2) model fine-tuned on the SST-2 part of the GLUE dataset. It was obtained from the "teacher" RoBERTa model by using task-specific knowledge distillation. Since it was fine-tuned on the SST-2, the final model is ready to be used in sentiment analysis tasks.

## Comparison to the original RoBERTa model:


The final distilled model was able to achieve 92% accuracy on the SST-2 dataset. Given the original RoBERTa achieves 94.8% accuracy on the same dataset with much more parameters (125M) and that this distilled version is nearly twice as fast as it is, the accuracy is impressive.


## Final Training Results after Hyperparameter Tuning
|       Epoch       |  Training Loss  |  Validation Loss  |   Accuracy   |
| ----------------- |   ------------  |     ---------     |  ----------  |
|1                  |    0.144000     |   0.379220        |   0.907110   |
|2                  |    0.108500     |   0.466671        |   0.911697   |
|3                  |    0.078600     |   0.359551        |   0.915138   |
|4                  |    0.057400     |   0.358214        |   0.920872   |


## Usage

To use the model from the 🤗/transformers library

```python
# !pip install transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("azizbarank/distilroberta-base-sst2-distilled")

model = AutoModelForSequenceClassification.from_pretrained("azizbarank/distilroberta-base-sst2-distilled")
```
## Notes: 
* The link to the model: https://huggingface.co/azizbarank/distilroberta-base-sst2-distilled
