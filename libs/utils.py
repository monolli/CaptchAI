"""
This is a speech recognition project for the Universidade Federal do ABC.
"""

import os
import re
from typing import List

import librosa as lr
import numpy as np
import pandas as pd


def zeroCrossings(data: np.ndarray) -> int:
    """Count how many times the amplitude of the wave crosses
        the zero threshold.

    Parameters
    ----------
    data : np.ndarray
        Audio time series. [shape=(n,)]

    Returns
    -------
    int
        Hown many times the wave crossed zero.

    """
    return sum(lr.zero_crossings(data, pad=False))


def normalize(data: np.ndarray, axis: int = 0, threshold: int = 100) -> np.ndarray:
    """Normalize the amplitude to fit between -1 and 1.

    Parameters
    ----------
    data : np.ndarray
        Audio time series. [shape=(n,)]
    axis : int
        Definition of the considered axis (0 = 'column' , 1 = 'row').
    threshold : int
        Minimum amplitude value to be considered in the normalization.

    Returns
    -------
    np.ndarray
        The normalized audio time series. [shape=(n,)]

    """
    return lr.util.normalize(data, axis=axis, threshold=threshold)


def myMfcc(data: np.ndarray, fs: int, n_mfcc: int) -> np.ndarray:
    """Calculate mfcc coefficients of each coefficient.

    Parameters
    ----------
    data : np.ndarray
        Audio time series. [shape=(n,)]
    fs : int
        The sampling ratio of the audio.
    n_mfcc : int
        The number of coefficients.

    Returns
    -------
    np.ndarray
        The MFCC sequence. [shape=(n_mfcc, t)]

    """
    return pd.DataFrame(lr.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, n_fft=512))


def cropAudio(path: str, interval: int) -> List[str]:
    """Crops the audio by character.

    Parameters
    ----------
    path : str
        Path of the .wav audio file containing 4 characters.
    interval : int
        Interval in seconds that contais each character.

    Returns
    -------
    List[str]
        List with each captured character and its sample rate and character.

    """
    # Loads the .wav audio file
    data, fs = lr.load(path, offset=.2)
    # Prepare to receive the cropped data
    data_list = []
    # Get the char from the filename
    character = list(re.search(r"([^\/\\\s]+)(?:\s\(\d+\))?\.wav", path)[1])
    # Crops the raw data in intervals
    for i, beg in enumerate(range(0, data.shape[0], fs * interval)):
        if i < 4:
            # Crop data
            c_data = data[beg:(beg + fs * interval)]
            data_list.append([c_data, fs, character[i]])
    return data_list


def buildDataFrame(path: str, out: str, raw_data: bool = False, mfcc: bool = True, mfcc_d: bool = True,
                   mfcc_dd: bool = True, z_crossings: bool = True, normalized: bool = True, pad: int = 44100) -> None:
    """Reads de audio files, treats the data and builds a data frame.

    Parameters
    ----------
    path : str
        Path of the directory containing the .wav files.
    out : str
        Path of the ouputfile containing the dataframe.
    raw_data : bool
        Boolean indicating if the raw data should be returned or not.
    mfcc : bool
        Boolean indicating if the MFCC should be calculated or not.
    mfcc_d : bool
        Boolean indicating if the MFCC' should be calculated or not.
    mfcc_dd : bool
        Boolean indicating if the MFCC" should be calculated or not.
    z_crossings : bool
        Boolean indicating if the "zero crossings" should be calculated or not.
    normalized : bool
        Boolean indicating if the data should be normalized or not.
    pad : int
        The size of the padding.

    Returns
    -------
    None
        Returns nothing.

    """
    # Initialize a dataframe to receive the features and the classes
    data = pd.DataFrame()
    # Iterate over the files of the given directory
    for filename in os.listdir(os.getcwd() + path):
        # Select only the .wav files and capture their filenames
        if re.search(r".+\.wav", filename):
            # Iterate over each character of the file
            for char in cropAudio(os.getcwd() + path + filename, 2):
                # Initialize an empty numpy array
                line = []
                # Check if the data is going to be normalized
                if normalized:
                    # Normalize the raw data
                    char[0] = normalize(char[0])
                # Check if the data is going to be padded
                if pad:
                    # Pad right with zero
                    char[0] = np.pad(char[0], (0, pad - len(char[0])),
                                     mode="constant")
                # Check if the threated raw data is going to be used
                if raw_data:
                    # Add the raw data in the dataframe
                    line = np.concatenate((line, char[0]), axis=None)
                # Check if the MFCC is going to be used
                if mfcc:
                    # Calculate de MFCC
                    mfccs = myMfcc(char[0], char[1], 15)
                    # Calculate de MFCC stats
                    mfcc_st = mfccs.assign(mean=mfccs.mean(axis=1),
                                           median=mfccs.median(axis=1),
                                           std=mfccs.std(axis=1),
                                           var=mfccs.var(axis=1),
                                           max=mfccs.max(axis=1),
                                           min=mfccs.min(axis=1)).iloc[:, -6:].to_numpy().flatten()
                    # Add the MFCC to the dataframe
                    line = np.concatenate((line, mfcc_st), axis=None)
                    # Check if the MFCC Delta is going to be used
                    if mfcc_d:
                        # Calculate de delta for the MFCC
                        mfccd = pd.DataFrame(lr.feature.delta(mfccs))
                        # Calculate de MFCC_D stats
                        mfccd_st = mfccd.assign(mean=mfccd.mean(axis=1),
                                                median=mfccd.median(axis=1),
                                                std=mfccd.std(axis=1),
                                                var=mfccd.var(axis=1),
                                                max=mfccd.max(axis=1),
                                                min=mfccd.min(axis=1)).iloc[:, -6:].to_numpy().flatten()
                        # Add the MFCC_D to the dataframe
                        line = np.concatenate((line, mfccd_st), axis=None)
                    # Check if the MFCC Delta Delta is going to be used
                    if mfcc_dd:
                        # Calculate de deltadelta for the MFCC
                        mfccd2 = pd.DataFrame(lr.feature.delta(mfccs, order=2))
                        # Calculate de MFCC_DD stats
                        mfccd2_st = mfccd2.assign(mean=mfccd2.mean(axis=1),
                                                  median=mfccd2.median(axis=1),
                                                  std=mfccd2.std(axis=1),
                                                  var=mfccd2.var(axis=1),
                                                  max=mfccd2.max(axis=1),
                                                  min=mfccd2.min(axis=1)).iloc[:, -6:].to_numpy().flatten()
                        # Add the MFCC_DD to the dataframe
                        line = np.concatenate((line, mfccd2_st), axis=None)
                # Check if the zero crossing rate is going to be used
                if z_crossings:
                    # Calculate the zero crossing rate for the audio segment
                    z_crossings = zeroCrossings(char[0])
                    # Add the zero crossing rate to the dataframe
                    line = np.concatenate((line, z_crossings), axis=None)
                # Add the class of the audio segment
                line = np.concatenate((line, char[2]), axis=None)
                # Append thesegment dataframe to the global dataframe
                data = data.append(pd.DataFrame(line.reshape(-1, len(line))),
                                   ignore_index=True)
    # Write the dataframe to a csv file
    with open(out, "w") as outfile:
        data.to_csv(outfile)
