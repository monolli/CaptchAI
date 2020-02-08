"""
This is a speech recognition project for the Universidade Federal do ABC.
"""

import os
import re

import librosa as lr
import librosa.display as disp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs import utils


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    data = pd.DataFrame()
    # Iterate over the files of the given directory
    i = 0
    for filename in os.listdir(os.getcwd() + "/data/training/"):
        # Select only the .wav files and capture their filenames
        if re.search(r".+\.wav", filename) and i < 5:
            # Iterate over each character of the file
            for char in utils.cropAudio(os.getcwd() + "/data/training/" + filename, 2):

                if char[2] == "m" and i < 5:
                    i += 1
                    # Initialize an empty numpy array
                    line = []
                    # Normalize the raw data
                    char[0] = utils.normalize(char[0])
                    # Check if the data is going to be padded
                    char[0] = np.pad(char[0], (0, 44100 - len(char[0])),
                                     mode="constant")
                    # Add the raw data in the dataframe
                    line = np.concatenate((line, char[0]), axis=None)
                    # Add the sample Rate
                    line = np.concatenate((line, char[1]), axis=None)
                    # Add the class of the audio segment
                    line = np.concatenate((line, char[2]), axis=None)
                    # Append the segment dataframe to the global dataframe
                    data = data.append(pd.DataFrame(line.reshape(-1, len(line))),
                                       ignore_index=True)
                else:
                    break
        else:
            break

    # Plot waves
    plt.figure(1)
    plt.figure(1).subplots_adjust(hspace=.5)

    plt.subplot(421)
    disp.waveplot(data.iloc[0, :-3].apply(pd.to_numeric).to_numpy(), sr=int(float(data.iloc[0, -2])))

    plt.subplot(422)
    disp.specshow(lr.feature.mfcc(y=data.iloc[0, :-3].apply(pd.to_numeric).to_numpy(),
                                  sr=int(float(data.iloc[0, -2])), n_mfcc=15, n_fft=512))

    plt.subplot(423)
    disp.waveplot(data.iloc[1, :-3].apply(pd.to_numeric).to_numpy(), sr=int(float(data.iloc[1, -2])))

    plt.subplot(424)
    disp.specshow(lr.feature.mfcc(y=data.iloc[1, :-3].apply(pd.to_numeric).to_numpy(),
                                  sr=int(float(data.iloc[1, -2])), n_mfcc=15, n_fft=512))

    plt.subplot(425)
    disp.waveplot(data.iloc[2, :-3].apply(pd.to_numeric).to_numpy(), sr=int(float(data.iloc[2, -2])))

    plt.subplot(426)
    disp.specshow(lr.feature.mfcc(y=data.iloc[2, :-3].apply(pd.to_numeric).to_numpy(),
                                  sr=int(float(data.iloc[2, -2])), n_mfcc=15, n_fft=512))

    plt.subplot(427)
    disp.waveplot(data.iloc[3, :-3].apply(pd.to_numeric).to_numpy(), sr=int(float(data.iloc[3, -2])))

    plt.subplot(428)
    disp.specshow(lr.feature.mfcc(y=data.iloc[3, :-3].apply(pd.to_numeric).to_numpy(),
                                  sr=int(float(data.iloc[3, -2])), n_mfcc=15, n_fft=512))

    plt.savefig("./graphs/m_mel_spectrogram.png")


if __name__ == "__main__":
    main()
