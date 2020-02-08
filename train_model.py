"""
This is a speech recognition project for the Universidade Federal do ABC.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    # Load training data
    with open("./train.csv") as data_file:
        train = pd.read_csv(data_file)

    # Load validation data
    with open("./valid.csv") as data_file:
        valid = pd.read_csv(data_file)

    # Train a random forest
    # !!!!The number os estimators is not optimized!!!!!
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
    # Return the most valuables attributes
    selected = SelectFromModel(clf, threshold="1.25*median")
    # Factorize the classes
    fact, char = pd.factorize(train.iloc[:, -1], sort=True)
    # Train with every attribute
    print("Fitting...")
    selected.fit(train.iloc[:, :-1], fact)
    # Train with the most relevant attributes
    print("Refitting...")
    clf.fit(train.iloc[:, :-1].loc[:, selected.get_support()], fact)

    """
    # Cross validation
    param_grid = [{"n_estimators": [100, 300, 500, 800, 1200],
                   "max_depth": [5, 8, 15, 25, 30],
                   "min_samples_split": [2, 5, 10, 15, 100],
                   "min_samples_leaf": [1, 2, 5, 10]}
                  ]

    grid_search = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(train.iloc[:, :-1], fact)

    print("Best params:")
    print()
    print(grid_search.best_params_)
    print()
    print("accuracy:")
    print()
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds,
                                 grid_search.cv_results_["params"]):
        print("\t%0.3f (+/-%0.03f) :: %r" % (mean, std * 2, params))
        print()

    # Apply the learned model
    pred = grid_search.best_estimator_.predict(valid.iloc[:, :-1])
    """

    pred = clf.predict(valid.iloc[:, :-1].loc[:, selected.get_support()])

    matrix = pd.crosstab(valid.iloc[:, -1], char[pred],
                         rownames=["Actual Char"], colnames=["Predicted Char"])

    print(matrix)
    print(accuracy_score(valid.iloc[:, -1], char[pred]))

    # Train the a model with all the data
    bigdata = train.append(valid, ignore_index=True)
    fact, char = pd.factorize(bigdata.iloc[:, -1], sort=True)

    print("Fitting the final model...")
    clf.fit(bigdata.iloc[:, :-1].loc[:, selected.get_support()], fact)
    # Save the final model
    print("Saving the model...")

    with open("./final_model.pkl", "wb") as model_file:
        joblib.dump(clf, model_file)
    with open("./features.pkl", "wb") as features_file:
        joblib.dump(selected.get_support(), features_file)

    ax = sns.heatmap(matrix, annot=True, cbar=False, square=True, fmt="d")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    with open("./graphs/confusion_matrix.png", "wb") as graph_file:
        plt.savefig(graph_file)

    # TODO REMOVE BACKGROUND NOISE
    # TODO FILTER FOR VOCAL WAVE LEGHT (MADE IT WORSE)
    # TODO TRIM SILENCE


if __name__ == "__main__":
    main()
