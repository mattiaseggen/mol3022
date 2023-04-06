import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


def read_dataset(file_path, max_sequence_length, input_col, target_col):
    """
    This function reads the dataset from file and inserts the desired input
    and target sequences into the desired columns.
    """
    df = pd.read_csv(file_path)
    print(df.shape)

    input_sequences, target_sequences = df[[input_col, target_col]][(
        df.len <= max_sequence_length) & (~df.has_nonstd_aa)].values.T

    return input_sequences, target_sequences


def preprocess_data(data, max_sequence_length, is_input=True):
    """
    This function creates tokens from the dataset and pads the sequences if
    they are shorter than the maximum sequence length.
    """
    if is_input is True:
        input_data = sequence2ngram(data)
        input_data, input_tokenizer = tokenize(input_data)
        input_data = pad_sequences(
            input_data, max_sequence_length, padding="post")
        print("Input data", input_data.shape)
        return input_data, input_tokenizer
    else:
        target_data, target_tokenizer = tokenize(data, char_level=True)
        target_data = pad_sequences(
            target_data, max_sequence_length, padding="post")
        target_data = to_categorical(target_data)
        print("Target data", target_data.shape)
        return target_data, target_tokenizer


def tokenize(data, char_level=False):
    tokenizer = Tokenizer(char_level=char_level)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)

    return data, tokenizer


def sequence2ngram(sequences, n=3):
    return np.array([[sequence[i:i+n] for i in range(len(sequence))]
                     for sequence in sequences], dtype=object)
