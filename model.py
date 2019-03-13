from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import CuDNNGRU
from keras.layers import Activation
from keras.layers import Bidirectional, Flatten


def create_network(network_input, n_vocab, mode, weights=None):
    sequence_length = network_input.shape[1]  # 100
    data_dim = network_input.shape[2]  # 1

    model = Sequential()
    if mode == "three_layer":
        print("Using a Three Layer LSTM model")
        # model.add(CuDNNLSTM(
        model.add(LSTM(
            256,
            input_shape=(sequence_length, data_dim),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        # model.add(CuDNNLSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        # model.add(CuDNNLSTM(256))
        model.add(LSTM(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))

    elif mode == "bidirectional":
        print("Using a Bidirectional LSTM model")
        # model.add(Bidirectional(CuDNNLSTM(256),
        model.add(Bidirectional(LSTM(256),
                                input_shape=(sequence_length, data_dim)
                                ))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))  # based on number of unique notes
        model.add(Dropout(0.3))

    elif mode == "stacked":
        print("Using a Stacked LSTM model")
        # model.add(CuDNNLSTM(
        model.add(LSTM(
            256,
            input_shape=(sequence_length, data_dim),
            return_sequences=True
        ))
        # model.add(CuDNNLSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        # model.add(CuDNNLSTM(256))
        model.add(LSTM(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
    elif mode == "gru":
        print("Using a three layer GRU model")
        # model.add(CuDNNGRU(
        model.add(GRU(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        # model.add(CuDNNGRU(256, return_sequences=True))
        model.add(GRU(256, return_sequences=True))
        model.add(Dropout(0.3))
        # model.add(CuDNNGRU(256))
        model.add(GRU(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
    else:
        print("ERROR: architecture doesnt exist")

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights:
        model.load_weights(weights)

    return model
