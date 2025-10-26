import sys

import requests

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from IPython import embed


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('f1_score: ', f1_score(y, yhat))
    print('mean(y): ', np.mean(y))
    print('mean(yhat): ', np.mean(yhat))


def build_simple_examples_string():
    example_string = """
####
Here is a disaster example:

Inquiry: Heavy accident on Interstate 10.
1

Here is a non-disaster example:

Inquiry: Good morning everybody.
0
####
"""

    # print(example_string)

    return example_string


def model(X_test, examples_string=""):
    yhat = []

    instruction = """You are a social media analyst. Your task is to classify a tweet after <<<>>> to be either about a disaster or about something else. Return '1' if you think the provided tweet is actually about a disaster. Return '0' if it's really about something else.
You will only respond with '1' or '0'. Do not provide any explanations or notes."""

    for i in range(len(X_test)):
        inquiry = X_test["text"].iloc[i]

        # 0.76 F1 score with the two generic examples (0.74 without)
        prompt = examples_string + """
<<<
Inquiry: {}
>>>""".format(inquiry)

        # print(instruction)
        # print(prompt)

        json_data = {
            "model": "mistral",
            "system": instruction,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 2, "seed": 42, "temperature": 0},
        }

        response = requests.post('http://localhost:11434/api/generate', json=json_data)
        yhat_item = response.json()["response"]
        # print(yhat_item)
        yhat_item = yhat_item.strip(' .')
        if yhat_item not in ['0', '1']:
            yhat_item = 0
            print("blabla")
        yhat_item = int(yhat_item)
        yhat.append(yhat_item)

    return np.array(yhat)


def main(args):
    np.random.seed(42)

    df_train = pd.read_csv("../train.csv")
    df_test = pd.read_csv("../test.csv")

    df_train.fillna("", inplace=True)
    df_test.fillna("", inplace=True)

    y = df_train["target"]
    print('mean(y): ', np.mean(y))
    X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=666)
    X_train["target"] = y_train

    examples_string = build_simple_examples_string()

    # validation
    yhat = model(X_test, examples_string)
    evaluation(y_test, yhat)

    yhat = model(df_test, examples_string)
    pd.concat([df_test["id"], pd.Series(yhat, name="target")], axis=1).to_csv("submission.csv", index=False)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
