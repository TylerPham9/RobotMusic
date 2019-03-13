""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
import os
import argparse
from music21 import instrument, note, stream, chord, converter
from model import create_network
import editdistance

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str,
                    help="Name of the dataset",
                    choices=["midi_songs", "Pop_Music_Midi",
                             "christmas_songs"])
parser.add_argument("--mode", "-m", type=str,
                    help="Type of RNN architecture",
                    choices=["three_layer", "bidirectional", "stacked",
                             "gru"])
args = parser.parse_args()

outputdir = "output/"

def generate(song_path, sequence_length):
    # If song name is 'random', use a random sequence
    """ Generate a piano midi file """
    # load the notes used to train the model
    with open(args.dataset + '/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequences_predict(notes,
                                                                pitchnames,
                                                                n_vocab)
    model = create_network(normalized_input, n_vocab, args.mode,
                           "weights/" + args.mode + "_" + args.dataset + ".hdf5")
    song_name = song_path.split("/")[-1]

    if song_name != "random.mid":
        # Get notes of input song
        song_notes = get_input_notes(song_path)

        # Create a processed midi of the song we want to predict
        create_midi(song_notes, outputdir + "full_" + song_name)

        # Get the sequence after 100 notes
        if sequence_length > len(song_notes):
            end = None
        else:
            end = 100 + sequence_length
        expected_song = song_notes[100:end]

        # Create a midi of the expected
        create_midi(expected_song, outputdir + "expected_" + song_name)
        song_input, _ = prepare_sequences_predict(song_notes, pitchnames,
                                                  n_vocab)
        prediction_output = generate_notes(model, song_input, pitchnames,
                                           n_vocab, sequence_length, False)
    else:
        prediction_output = generate_notes(model, network_input, pitchnames,
                                           n_vocab, sequence_length, True)

    create_midi(prediction_output, outputdir + "prediction_" + song_name)


def prepare_sequences_predict(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        new_sequence = []
        for char in sequence_in:
            if char in note_to_int:
                new_sequence.append(note_to_int[char])
            else:
                new_sequence.append(0)
        network_input.append(new_sequence)

        if sequence_out in note_to_int:
            output.append(note_to_int[sequence_out])
        else:
            output.append(0)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input,
                                     (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def generate_notes(model, network_input, pitchnames, n_vocab,
                   sequence_length, random):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    if random:
        start = numpy.random.randint(0, len(network_input) - 1)
    else:
        start = 0

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # Get the start of the song
    pattern = network_input[start]
    prediction_output = []

    # generate notes from sequence_length
    for note_index in range(sequence_length):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output, name):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=(name))


def get_input_notes(filepath):
    notes = []
    midi = converter.parse(filepath)

    print("Parsing %s" % filepath)

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def compare_songs(expected, predicted):
    # Convert songs to the sequence of chords/notes
    expected_notes  = get_input_notes(expected)
    predicted_notes = get_input_notes(predicted)

    total_notes = len(expected_notes)
    print("total number of notes: %i" % total_notes)
    # Compare the same length of songs
    predicted_notes = predicted_notes[:total_notes]
    distance = editdistance.eval(expected_notes, predicted_notes)
    print("Distance: %i" % distance)
    accuracy = (total_notes-distance)/total_notes
    print("Accuracy: %f" % accuracy)
    return accuracy


if __name__ == '__main__':
    # song = "wondrlan.mid"
    # generate(song, 300)
    # # generate("random.mid", 300)
    # compare_songs("expected_" + song, "prediction_" + song)

    testdir = "test"
    scores = []
    for filename in os.listdir(testdir):
        song = testdir + "/" + filename
        generate(song, 300)
        score = compare_songs(outputdir + "expected_" + filename,
                              outputdir + "prediction_" + filename)
        scores.append(score)

    mean_score = sum(scores)/len(scores)
    print(mean_score)

