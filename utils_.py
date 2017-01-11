from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Convolution2D, Permute, Reshape, Merge, Input, Convolution1D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, SGD
from keras.engine import Model
from IPython import embed

# %% Notes and Chord vocabulary
NOTES = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B', 'N']
CHORDS = ['maj', 'min', '7', 'min7', 'maj7', '5', '1', '', 'maj(9)', 'maj6', 'sus4', 'sus7', 'sus9', '7(#9)', 'min9']

#%% Util functions
def openWithoutFirstCol(inputF):
    with open(inputF, "r") as csvfile:
        data_csv = csvfile.read().splitlines()
    csvfile.close()

    j=0
    Nfeatures = 24
    frOn_ = np.zeros((len(data_csv), 1))
    one_hot = np.zeros((len(data_csv), Nfeatures), dtype=float)
    for row in data_csv:
        line = row.split(',')
        frOn_[j] = line.pop(0)
        one_hot[j, :] = np.array(line)
        j +=1

    return frOn_, one_hot


def csvOneHotEncode(frOn_, data_csv_):
    Nfeatures = 24
    one_hot = np.zeros((len(frOn_), Nfeatures), dtype=int)
    for r in range(len(frOn_)):
        prob = np.array(data_csv_[r].split(','), dtype=float)
        prob = prob[1:] #Elimino el onset time
        one_hot[r,:] = prob
        # if (np.sum(prob) != 0):  # puede ser que haya filas de solo ceros
        #     one_hot[r, prob.argmax()] = 1
    return one_hot


def labOneHotEncode(framei, data_lab_, i, dictionaryNotes, dictionaryChords, counter):
    cols = len(dictionaryNotes) + len(dictionaryChords)
    one_hot = np.zeros((len(framei), cols), dtype=int)
    notes = np.zeros((len(framei), 1), dtype=int)
    chords = np.zeros((len(framei), 1), dtype=int)

    chords_bin = []
    timestep_bin = []

    print 'File ', i+1, '/ 890', '...'

    note_i=[]
    chord_i=[]
    i_frame=0

    for r in range(len(data_lab_)-1):  # recorre longitud del fichero lab
        # Get chord, onset and offset times
        split = data_lab_[r].split()
        ch = split.pop()
        ons = float(split.pop(0))
        offs = float(split.pop())
        # Get only the root note
        rootC = ch.split(':')[0]
        chord = ch.split(':')[-1]
        # Check if it belongs to the vocabulary (e.g. /0033)
        if (dictionaryNotes.has_key(rootC)):
            note_i.append(rootC)
        else:
            if rootC == 'Cb':
                note_i.append('B')
                #print ('---- Cb replaced ')
            elif rootC == 'Fb':
                note_i.append('E')
                #print ('---- Fb replaced ')
            else:
                note_i.append('N')
            #print '\nChord ', rootC, ' has been replaced by "N"'
        # counter[0,dictionaryNotes[note_i[r]]] += 1  # para saber cuantas notaciones de cada acorde hay en los ficheros de los expertos (.lab)

        if (dictionaryChords.has_key(chord)):
            chord_i.append(chord)
        else:
            chord_i.append('')

        while ((float(framei[i_frame]) < offs) and (i_frame < (len(framei)-1))):
            if (float(framei[i_frame])-ons >= -1e-10):
                one_hot[i_frame, dictionaryNotes[note_i[r]]] = 1
                notes[i_frame] = dictionaryNotes[note_i[r]]

                chords[i_frame] = dictionaryChords[chord_i[r]]

                one_hot[i_frame, (dictionaryChords[chord_i[r]] + len(dictionaryNotes))] = 1

            if (i_frame>0 and notes[i_frame] != notes[i_frame - 1]):
                chords_bin.append(notes[i_frame-1])
                timestep_bin.append( float(framei[i_frame]) )
            i_frame += 1

    return notes, chords, one_hot, timestep_bin


def adjustDimension(length_, notes_, chords_, one_hot_csv_, one_hot_lab_, dictionaryNotes, dictionaryChords):
    one_hot_equal = []
    one_hot_equal_target = []
    note_equal = []
    ch_equal = []
    #l_max_ = np.max(length_)
    l_mean_ = int(np.round(np.mean(length_)))
    cols = len(dictionaryNotes) + len(dictionaryChords)

    for i_ in range(len(length_)):
        dif = int(l_mean_ - length_[i_])
        if dif > 0:
            #frOn[i_, 0] = np.append(frOn[i_, 0], np.zeros((dif, 1)))
            one_hot_equal.append(np.append(one_hot_csv_[i_, 0], np.zeros((dif, 24), dtype=int), axis=0))
            one_hot_equal_target.append(
                np.append(one_hot_lab_[i_, 0], np.zeros((dif, cols), dtype=int), axis=0))
            note_equal.append(np.append(notes_[i_, 0], np.zeros((dif, 1), dtype=int), axis=0))
            ch_equal.append(np.append(chords_[i_, 0], np.zeros((dif, 1), dtype=int), axis=0))
        else:
            one_hot_equal.append(one_hot_csv_[i_, 0][0:l_mean_])
            one_hot_equal_target.append(one_hot_lab_[i_, 0][0:l_mean_])
            note_equal.append(notes_[i_, 0][0:l_mean_])
            ch_equal.append(chords_[i_, 0][0:l_mean_])

    data_ = np.reshape(one_hot_equal, (len(one_hot_equal), 1, l_mean_, 24))
    target_ = np.reshape(one_hot_equal_target, (len(one_hot_equal_target), l_mean_, cols))
    target_note = np.reshape(note_equal, (len(note_equal), l_mean_))
    target_chord = np.reshape(ch_equal, (len(ch_equal), l_mean_))

    return data_, target_, target_note, target_chord


def save_to_file_matrix(matrix_notes, matrix_chords, path):
    file = open(path, 'w')
    file.write('Notes confusion matrix:\n' + str(matrix_notes) + '\n\n')
    file.write('Chords confusion matrix:\n' + str(matrix_chords))
    file.close()
    return


def save_to_file(note, chord, frame, path):
    file = open(path, 'w')
    t_on = frame[0,0]
    for tk in range(len(frame)-2):
        if (str(NOTES[note[tk]]) != str(NOTES[note[tk+1]])) or (str(CHORDS[chord[tk]]) != str(CHORDS[chord[tk+1]])):
            if CHORDS[chord[tk]] == '' or NOTES[note[tk]] == 'N':
                chord_label = str(NOTES[note[tk]])
                file.write(str(t_on) + ' ' + str(frame[tk+1,0]) + ' ' + str(chord_label) + '\n')
            else:
                chord_label = str(NOTES[note[tk]]) + ':' + str(CHORDS[chord[tk]])
                file.write(str(t_on) + ' ' + str(frame[tk+1,0]) + ' ' + str(chord_label) + '\n')
            t_on = frame[tk+1,0]
    file.close()
    return


def baseline_model(nb_filters, pooling, dim, data_, target_train_, loading='False', path=""):
    nb_input_time = int(data_.shape[2]/2)
    nb_channels = int(nb_filters)  # of feature maps in the last conv layer
    fnn_init = 'he_uniform'

    input = Input(shape=(data_.shape[1], data_.shape[2], data_.shape[3]), name='input_part')

    out1 = Convolution2D(
        nb_filter=nb_filters,
        nb_row=3,
        nb_col=dim,
        dim_ordering='th',
        init=fnn_init,
        border_mode='same',
        activation='relu'
    )(input)

    out = Convolution2D(
        nb_filter=nb_filters*2,
        nb_row=3,
        nb_col=dim,
        dim_ordering='th',
        init=fnn_init,
        border_mode='same',
        activation='relu'
    )(out1)

    # move the timesteps to the first axis and the number of channels to the second axis.
    out=Permute((2, 1, 3))(out)

    out = MaxPooling2D(
        pool_size=(2,2),
        strides=None,
        border_mode='same'
    )(out)

    # condense the last two axes (number of channels * outputs per channel):
    nb_input_freq = nb_channels*data_.shape[3]
    out = Reshape((nb_input_time, nb_input_freq))(out)

    output = TimeDistributed(Dense(output_dim=target_train_.shape[2], init=fnn_init, activation='softmax'))(out)

    m = Model(input, output)

    if loading == 'True':
        m.load_weights(path)

    # Compile model
    ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, decay=0.0)
    m.compile(loss='binary_crossentropy', optimizer=ADAM, metrics=['accuracy'])
    return m



def complex_model(nb_filters, pooling, dim, data_, target_train_, loading='False', path=""):
    nb_input_time = int(data_.shape[2] / pooling)
    nb_channels = int(nb_filters / 2)  # of feature maps in the last conv layer
    fnn_init = 'he_uniform'

    input = Input(shape=(data_.shape[1], data_.shape[2], data_.shape[3]), name='input_part')

    out = Convolution2D(
        nb_filter=nb_filters,
        nb_row=dim,
        nb_col=dim,
        dim_ordering='th',
        init=fnn_init,
        border_mode='same',
        activation='relu'
    )(input)

    out = MaxPooling2D(
        pool_size=(2, 2),
        strides=None,
        border_mode='same'
    )(out)  # similar results but this way is faster implementation

    # out = Convolution2D(
    #     nb_filter=int(nb_filters/2),
    #     nb_row=dim,
    #     nb_col=dim,
    #     dim_ordering='th',
    #     init=fnn_init,
    #     border_mode='same',
    #     activation='relu'
    # )(out)

    # move the timesteps to the first axis and the number of channels to the second axis:
    out = Permute((2, 1, 3))(out)
    # condense the last two axes (number of channels * outputs per channel):
    nb_input_freq = nb_channels * data_.shape[3]
    out = Reshape((nb_input_time, nb_input_freq))(out)

    for k in range(int(np.log2(nb_input_freq+1))):
        if k!=0 and k%2 == 0:
            out = Convolution1D(
                nb_filter=int(nb_input_freq /k),
                filter_length=dim,
                init=fnn_init,
                border_mode='same',
                activation='relu'
            )(out)

    output = TimeDistributed(Dense(output_dim=target_train_.shape[2], init=fnn_init, activation='softmax'))(out)

    m = Model(input, output)

    if loading == 'True':
        m.load_weights(path)

    # Compile model
    ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, decay=0.0)
    m.compile(loss='binary_crossentropy', optimizer=ADAM, metrics=['accuracy'])
    return m



