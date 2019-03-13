# train.py
""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import os
import argparse
from music21 import converter, instrument, note, chord

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback
from model import create_network

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str,
                    help="Name of the dataset",
                    choices=["midi_songs", "Pop_Music_Midi",
                             "christmas_songs"])
parser.add_argument("--mode", "-m", type=str,
                    help="Type of RNN architecture",
                    choices=["three_layer", "bidirectional", "stacked",
                             "gru"])
parser.add_argument("--epochs", "-e", type=int,
                    help="Number of epochs to train",
                    default=10)

args = parser.parse_args()

train_folder = args.dataset + "_train"

if not os.path.exists(train_folder):
    os.mkdir(train_folder)
    print("Directory ", train_folder,  " Created ")
else:    
    print("Directory ", train_folder,  " already exists")


def train_network():
    """ Train a Neural Network to generate music """
    
    # check if the notes file already exists
    if os.path.isfile(args.dataset + "/notes"):
        with open(args.dataset + '/notes', 'rb') as filepath:
            notes = pickle.load(filepath)
    else:
        # if not create the notes and store in the dataset folder
        notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab, args.mode)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(args.dataset + "/*.mid*"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(args.dataset + '/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))  


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = train_folder + "/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    
    
    history = LossHistory()
    history = model.fit(network_input, network_output, epochs=args.epochs,
                        batch_size=64, callbacks=callbacks_list)
    history_filepath = train_folder + "/" + args.dataset + '_history.pkl'
    with open(history_filepath, 'wb') as f:
        pickle.dump(history.history, f)
    print("History: {}".format(history.history))
    print("History saved at {}".format(history_filepath))
    
    list_of_files = glob.glob(train_folder+'/weights*')
    latest_weight_file = max(list_of_files, key=os.path.getctime)

    print("Weights: %s" % latest_weight_file)

if __name__ == '__main__':
    train_network()
