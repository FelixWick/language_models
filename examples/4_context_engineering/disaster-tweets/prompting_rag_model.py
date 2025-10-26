import sys

import ollama
import chromadb

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from IPython import embed


use_rag = True
use_closest_label = False
create_collection = True


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def generate(prompt, data=None):
    # generate a response combining the prompt and data retrieved

    instruction = """You are a social media analyst. Your task is to classify a tweet after <<<>>> to be either about a disaster or about something else. Return 1 if the provided tweet is actually about a disaster or 0 if it is about something else.
You will only respond with 1 or 0. Do not provide any explanations or notes."""

    if data:
        text = data[:-2]
        label = data[-1]
        example_string = """
####
Here is a similar example:

Inquiry: {}
{}
####
""".format(text, label)
    else:
        example_string = ""

    # 0.74 F1 score with closest example (0.74 without example, 0.76 with two generic ones)
    prompt = example_string + """
<<<
Inquiry: {}
>>>""".format(prompt)

    # print(instruction)
    # print(prompt)

    output = ollama.generate(
        model="mistral",
        system=instruction,
        prompt=prompt,
        options={"num_predict": 3, "seed": 42, "temperature": 0},
    )
    yhat = output['response']

    yhat = yhat.strip(' .')

    try:
        yhat = int(yhat)
    except ValueError:
        yhat = 0
        print("not int")

    if yhat not in [0, 1]:
        yhat = 0
        print("not 0 or 1")

    # print(prompt)
    # print(yhat)

    return yhat


def retrieve(collection, embedding_model, prompt):
    # generate an embedding for the prompt and retrieve the most relevant example

    response = ollama.embeddings(
        prompt=prompt,
        model=embedding_model
    )
    query_embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    closest_sample = results['documents'][0][0]

    return closest_sample


def create_embeddings(collection, examples, embedding_model):
    # store each example in a vector embedding database

    for i, e in enumerate(examples):
        response = ollama.embeddings(model=embedding_model, prompt=e)
        embedding = response["embedding"]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[e]
        )


def retrieve_closest_train_sample(X_test, collection, embedding_model):
    closest_train_samples = []
    for prompt in X_test["text"].values:
        closest_sample = retrieve(collection, embedding_model, prompt)
        # print(closest_sample)
        closest_train_samples.append(closest_sample)
    return closest_train_samples


def generate_yhat(X_test, closest_train_samples=None):
    yhat = []
    for i in range(len(X_test)):
        if closest_train_samples:
            yhat.append(generate(X_test["text"].values[i], closest_train_samples[i]))
        else:
            yhat.append(generate(X_test["text"].values[i]))
    return yhat


def main(args):
    np.random.seed(42)

    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    df_train.fillna("", inplace=True)
    df_test.fillna("", inplace=True)

    y = df_train["target"]
    print('mean(y): ', np.mean(y))
    X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=666)

    embedding_model = "all-minilm"
    client = chromadb.PersistentClient(path="disaster-tweets-train-embeddings")

    examples = []
    for i in range(len(X_train)):
        examples.append("{} {}".format(X_train["text"].values[i], y_train.values[i]))

    if create_collection:
        client.delete_collection(name="train-valid")
        collection = client.create_collection(name="train-valid")
        create_embeddings(collection, examples, embedding_model)
    else:
        collection = client.get_collection(name="train-valid")

    if use_rag:
        closest_train_samples = retrieve_closest_train_sample(X_test, collection, embedding_model)
        if use_closest_label: # 0.73 F1 score
            yhat = []
            for i in closest_train_samples:
                yhat.append(int(i[-1]))
        else:
            yhat = generate_yhat(X_test, closest_train_samples)
    else:
        yhat = generate_yhat(X_test)

    evaluation(y_test.to_numpy(), np.array(yhat))

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
