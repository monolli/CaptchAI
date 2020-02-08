"""
This is a speech recognition project for the Universidade Federal do ABC.
"""

import os

from libs import utils


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    if not os.path.isfile("train.csv"):
        print("Wait a second, building the training dataset!!!")
        print("This is only required in the first execution.")
        utils.buildDataFrame("/data/training/", "./train.csv")
        print("Done =)")
    if not os.path.isfile("valid.csv"):
        print("Wait a second, building the validation dataset!!!")
        print("This is only required in the first execution.")
        utils.buildDataFrame("/data/validation/", "./valid.csv")
        print("Done =)")
    if not os.path.isfile("test.csv"):
        print("Wait a second, building the test dataset!!!")
        print("This is only required in the first execution.")
        utils.buildDataFrame("/data/test/", "./test.csv")
        print("Done =)")


if __name__ == "__main__":
    main()
