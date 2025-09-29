import sys

import torch
# import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification
from datasets import Dataset

import evaluate

# from peft import LoraConfig, get_peft_model, TaskType

from IPython import embed


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def finetuning(train_data, val_data, tokenizer):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 2,
    ).to(device)
    print(model)

    # only 0.75 F1 score without finetuning pre-trained layers
    # 0.78 with finetuning last ffn block of pre-trained layers only
    # 0.83 with full finetuning
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    # for param in model.base_model.transformer.layer[5].ffn.parameters():
    #     param.requires_grad = True
    # for param in model.base_model.transformer.layer[5].output_layer_norm.parameters():
    #     param.requires_grad = True

    print(sum(p.numel() for p in model.base_model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["q_lin", "k_lin", "v_lin"],
    #     bias='none',
    #     task_type=TaskType.SEQ_CLS
    # )
    # peft_model = get_peft_model(model, lora_config)
    # print(sum(p.numel() for p in peft_model.parameters() if p.requires_grad))
    # 0.79 with LoRA finetuning

    train_data = train_data.map(lambda samples: tokenizer(samples["text"], max_length=300, padding="max_length", truncation=True), batched=True)
    val_data = val_data.map(lambda samples: tokenizer(samples["text"], max_length=300, padding="max_length", truncation=True), batched=True)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
            output_dir="outputs",
            num_train_epochs=2,
            evaluation_strategy="epoch"
        )

    trainer = Trainer(
        model=model,
        # model=peft_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        compute_metrics=compute_metrics
    )
    trainer.train()

    model.save_pretrained("outputs")
    # peft_model.save_pretrained("outputs")


def predict(data, tokenizer, model):
    model.eval()

    yhats = np.array([])
    for samples in DataLoader(data, batch_size=32):
        input_ids = tokenizer(samples["text"], max_length=300, padding="max_length", truncation=True, return_tensors="pt")
        outputs = model(**input_ids.to(device))[0].cpu().detach().numpy()
        # outputs = F.softmax(model(**input_ids.to(device))[0], dim=1).cpu().detach().numpy()
        # For language outputs, one would often sample from the softmax probabilities.
        yhats = np.concatenate((yhats, np.argmax(outputs, axis=1)))

    return yhats


def main(args):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    df_train_full = df_train_full[["text", "target"]]
    df_test = df_test[["text", "id"]]

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    train_data = Dataset.from_pandas(df_train)
    train_data_full = Dataset.from_pandas(df_train_full)
    val_data = Dataset.from_pandas(df_val)
    test_data = Dataset.from_pandas(df_test)

    train_data = train_data.rename_column("target", "label")
    train_data_full = train_data_full.rename_column("target", "label")
    val_data = val_data.rename_column("target", "label")
    train_data = train_data.remove_columns(['__index_level_0__'])
    val_data = val_data.remove_columns(['__index_level_0__'])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # validation
    finetuning(train_data, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs").to(device)

    yhat_train_llm = predict(train_data, tokenizer, model)
    evaluation(np.array(train_data["label"]), yhat_train_llm)

    yhat_val_llm = predict(val_data, tokenizer, model)
    evaluation(np.array(val_data["label"]), yhat_val_llm)

    # test
    finetuning(train_data_full, val_data, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("outputs").to(device)

    yhat_train_full_llm = predict(train_data_full, tokenizer, model)
    evaluation(np.array(train_data_full["label"]), yhat_train_full_llm)

    yhat_test = predict(test_data, tokenizer, model)
    pd.concat([df_test["id"], pd.Series(yhat_test, name="target")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
