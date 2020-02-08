"""
This is a speech recognition project for the Universidade Federal do ABC.
"""

import joblib
import pandas as pd


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    with open("./test.csv") as data_file:
        test = pd.read_csv(data_file)
    # Load model
    clf = joblib.load("./final_model.pkl")

    # Load features
    features = joblib.load("./features.pkl")

    # Factorize the classes
    fact, char = pd.factorize(test.iloc[:, -1], sort=True)

    # Predict the class
    pred = clf.predict(test.iloc[:, :-1].loc[:, features])

    hit = 0
    miss = 0

    for i in range(0, len(char[pred]), 4):
        if (char[pred][i] == test.iloc[i, -1] and char[pred][i + 1] == test.iloc[i + 1, -1] and
           char[pred][i + 2] == test.iloc[i + 2, -1] and char[pred][i + 3] == test.iloc[i + 3, -1]):
            hit += 1
        else:
            miss += 1

    print("Hits: " + str(hit))
    print("Misses: " + str(miss))
    if miss == 0:
        print("Accuracy(4 characters): 1")
    else:
        print("Accuracy(4 characters): " + str(hit / (miss + hit)))

    matrix = pd.crosstab(test.iloc[:, -1], char[pred],
                         rownames=["Actual Char"], colnames=["Predicted Char"])

    print(matrix)


if __name__ == "__main__":
    main()
