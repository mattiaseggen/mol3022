import os
import json
import io
from keras.layers import Dense, LSTM, Embedding
import tensorflow as tf
from setup_data import read_dataset, preprocess_data
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.models import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split


def train(model, input_data, target_data, acc_metric,
          batch_size, loss_metric, max_epochs=256):

    dir = "model/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    model.compile(optimizer="rmsprop", loss=loss_metric,
                  metrics=["accuracy", acc_metric])

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, target_data, test_size=.4, random_state=0)

    # train the model
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=max_epochs, verbose=1)

    model.save(f'{dir}')
    # evaluate the model
    # model.evaluate(x_test, y_test, verbose=1)


def execute():
    data_set = './data/2018-06-06-ss.cleaned.csv'
    SEQUENCE_LENGTH = 128
    MAX_EPOCHS = 5
    BATCH_SIZE = 128
    LOSS_FUNCTION = "categorical_crossentropy"
    INPUT_TOKENIZER_PATH = "tokenizer/input_tokenizer.json"
    TARGET_TOKENIZER_PATH = "tokenizer/target_tokenizer.json"

    input_sequences, target_sequences = read_dataset(
        data_set, SEQUENCE_LENGTH, input_col='seq', target_col='sst3')

    input_data, input_tokenizer = preprocess_data(
        input_sequences, SEQUENCE_LENGTH, is_input=True)

    target_data, target_tokenizer = preprocess_data(
        target_sequences, SEQUENCE_LENGTH, is_input=False)

    input_dimension = len(input_tokenizer.word_index) + 1
    output_dimension = len(target_tokenizer.word_index) + 1

    print(input_dimension, output_dimension)

    save_tokenizer(input_tokenizer, path=INPUT_TOKENIZER_PATH)
    save_tokenizer(target_tokenizer, path=TARGET_TOKENIZER_PATH)

    model = get_model(input_dimension, output_dimension)
    model.summary()

    train(model, input_data, target_data, acc_metric=q3_acc,
          batch_size=BATCH_SIZE, loss_metric=LOSS_FUNCTION,
          max_epochs=MAX_EPOCHS)


def get_model(num_words, num_tags, hidden_layer_size=64,
              dropout=0.1, layers=1):
    OUTPUT_DIM = 128
    ACTIVATION_FUNCTION = "softmax"

    model = Sequential()
    model.add(Embedding(input_dim=num_words,
              output_dim=OUTPUT_DIM))  # Input length missing perhaps
    model.add(Bidirectional(
        LSTM(units=hidden_layer_size, return_sequences=True,
             recurrent_dropout=dropout)))
    model.add(TimeDistributed(Dense(num_tags, activation=ACTIVATION_FUNCTION)))

    return model


def q3_acc(y_true, y_pred):
    y = K.argmax(y_true, axis=-1)
    y_ = K.argmax(y_pred, axis=-1)
    mask = K.cast(K.greater(y, 0), K.floatx())
    matches = K.cast(K.equal(tf.boolean_mask(y, mask),
                     tf.boolean_mask(y_, mask)), K.floatx())
    return K.sum(matches) / K.maximum(K.sum(mask), 1)


def save_tokenizer(tokenizer, path):
    dir = os.path.dirname(path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    tokenizer_json = tokenizer.to_json()
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


if __name__ == '__main__':
    execute()
