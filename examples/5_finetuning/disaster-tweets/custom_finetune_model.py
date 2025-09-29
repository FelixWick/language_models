import sys

import torch
# import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DistilBertModel

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


class DistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        embedding_dim = 20
        self.keyword_embedding = nn.Embedding(222, embedding_dim)

        D_in, D_out = 768 + embedding_dim, 2
        self.pre_classifier = nn.Linear(D_in, D_in)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(D_in, D_out)

    def forward(self, input_ids, attention_mask, train_keyword):
        base_output = self.base_model(input_ids, attention_mask)
        hidden_state = base_output[0]
        # input to pooler
        cls_embedding = hidden_state[:, 0]
        # idea behind:
        # The output from the encoder is a matrix where each row is an enriched representation of
        # the corresponding token from the input sequence. However, for the task of sequence
        # classification, what is needed is a vector representation that captures the meaning of
        # the entire input sequence (not just one token). This could be achieved by taking the
        # element-wise mean or maximum of all the token representations. Here, we use a special
        # classification token [CLS] instead. It is placed at the very beginning of each sequence
        # and was developed for the pre-training phase with next-sentence prediction in mind,
        # intended to contain a contextualized, high-level representation of the entire sequence
        # (in its final state), what makes it a good candidate to be pooled for sequence
        # classification. Rationale: Although nothing makes [CLS] a good sentence representation in
        # the original pre-trained model (Every token is a weighted aggregate of the whole
        # sentence.), once you fine-tune it (e.g., for sentence classification of some sorts) you
        # are specifically training it to become a good sentence representation.

        X_keyword_embed = self.keyword_embedding(train_keyword.type(torch.LongTensor).to(device))
        X = torch.cat([cls_embedding, X_keyword_embed], dim=1)

        X = self.pre_classifier(X)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        logits = self.classifier(X)
        return logits


def fit(epochs, model, optimizer, train_dl):
    loss_func = nn.CrossEntropyLoss()

    # loop over epochs
    for epoch in range(epochs):
        model.train()

        # loop over mini-batches
        for input_ids, attention_mask, labels, keyword in train_dl:
            y_hat = model(input_ids, attention_mask, keyword)

            loss = loss_func(y_hat, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # print(loss.item())

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    print('Finished training')

    return model


def tokenization(tokenizer, text_data):
    encoded_corpus = tokenizer(
        text_data,
        max_length=300,
        padding="max_length",
        truncation=True,
        return_attention_mask=True
        )
    input_ids = torch.tensor(encoded_corpus['input_ids'])
    attention_mask = torch.tensor(encoded_corpus['attention_mask'])

    return input_ids, attention_mask


def predict(model, dataloader):
    model.eval()

    yhats = np.array([])
    for input_ids, attention_mask, keyword in dataloader:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, keyword).cpu().detach().numpy()
            # outputs = F.softmax(model(input_ids, attention_mask, keyword), dim=1).cpu().detach().numpy()
            # For language outputs, one would often sample from the softmax probabilities.
            yhats = np.concatenate((yhats, np.argmax(outputs, axis=1)))

    return yhats


def main(args):
    np.random.seed(42)
    torch.manual_seed(666)

    df_train_full = pd.read_csv("../train.csv")

    df_train_full.fillna("", inplace=True)

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # finetuning
    input_ids_train, attention_mask_train = tokenization(tokenizer, df_train["text"].tolist())

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    train_keyword = pd.DataFrame({"keyword": df_train["keyword"]})
    train_keyword = torch.tensor(enc.fit_transform(train_keyword)).view(-1).float().to(device)

    model = DistilBertClassifier()
    print(model)

    labels = torch.tensor(df_train["target"].tolist()).to(device)

    train_dataset = TensorDataset(input_ids_train.to(device), attention_mask_train.to(device), labels, train_keyword)
    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    # optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 2
    trained_model = fit(epochs, model.to(device), optimizer, train_dl)

    torch.save(trained_model, "outputs/model.pth")

    # inference
    input_ids_test, attention_mask_test = tokenization(tokenizer, df_val["text"].tolist())

    test_keyword = pd.DataFrame({"keyword": df_val["keyword"]})
    test_keyword = torch.tensor(enc.transform(test_keyword)).view(-1).float().to(device)

    test_dataset = TensorDataset(input_ids_test.to(device), attention_mask_test.to(device), test_keyword)
    test_dl = DataLoader(test_dataset, batch_size=128)

    inference_model = torch.load("outputs/model.pth")
    yhat = predict(inference_model, test_dl)
    evaluation(df_val["target"].values, yhat)
    # F1 score 0.82 (no improvement compared to finetuning without keyword)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
