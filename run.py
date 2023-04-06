import sys
import json
import numpy as np
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from tensorflow import keras
import tensorflow as tf


def main(argv):
    # Take a single sequence as input argument
    if len(sys.argv) != 2:
        print("Expected a protein sequence as argument")
        sys.exit(1)

    input = sys.argv[1]

    prediction = predict(input)
    print(f"Input:\t\t{input}")
    print(f"Prediction:\t{prediction.upper()}")


def get_ngrams(sequences, n=3):
    num_sequences = len(sequences)
    ngrams = np.empty(num_sequences, dtype=object)
    for s in range(num_sequences):
        ngrams[s] = ([sequences[s][i:i+n] for i in range(len(sequences[s]))])
    return ngrams


def predict(sequence):
    SEQUENCE_LENGTH = 128
    dir = "model/"
    with open("tokenizer/input_tokenizer.json") as input_t:
        tokenizer_json = json.load(input_t)
        input_tokenizer = tokenizer_from_json(tokenizer_json)

    with open("tokenizer/target_tokenizer.json") as target_t:
        tokenizer_json = json.load(target_t)
        target_tokenizer = tokenizer_from_json(tokenizer_json)

    model = keras.models.load_model(f"{dir}", custom_objects={
                                    "f1": f1}, compile=False)
    model.compile(metrics=["accuracy", f1])

    input_sequences = [sequence]
    input_ngrams = get_ngrams(input_sequences)

    input_data = input_tokenizer.texts_to_sequences(input_ngrams)
    input_data = pad_sequences(
        input_data, maxlen=SEQUENCE_LENGTH, padding="post")
    result = model.predict(input_data)
    prediction = tf.argmax(result, axis=-1)

    prediction = target_tokenizer.sequences_to_texts(prediction.numpy())

    return prediction[0].replace(" ", "")


def to_string(sequence, index):
    indices = np.argmax(sequence, axis=-1)
    non_zero_indices = np.nonzero(indices)
    seq = "".join([index[i] for i in indices[non_zero_indices]])
    return seq


def f1(y_true, y_pred):
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
