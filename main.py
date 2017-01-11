# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:33:43 2016

The project outcome consists in estimating a sequence of chords from audio music recordings using deep neural networks
as classifiers. The harmonic structure of different audio tracks from the Billaboard dataset (McGill University)
were extracted. This task required the estimation of a sequence of chords which was as precise as possible,
which included the full characterisation of chords – root, quality, and bass note – as well as their chronological
order, including specific onset times and durations. Some basic knowledge about music allowed me to enhance the results,
(e.g. taking into account that Cb = B and Fb = E)

To evaluate the quality of the automatic transcriptions, each one was compared to ground truth previously created by
a human annotator.

@author: lauracabello
"""

from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from sklearn.metrics import confusion_matrix
from time import time
from utils_ import *
import csv2txt
import config.nn_config as nn_config


# #%% Create dictionary with Notes and Chord vocabulary
dictionaryNotes = {'C':0, 'C#': 1, 'Db':2, 'D':3, 'D#':4, 'Eb':5, 'E':6, 'F':7, 'F#':8, 'Gb':9, 'G':10, 'G#':11,
                   'Ab':12, 'A':13, 'A#':14, 'Bb':15, 'B':16, 'N':17}

dictionaryChords = {'maj':0, 'min':1, '7':2, 'min7':3, 'maj7':4, '5':5, '1':6, '':7, 'maj(9)':8, 'maj6':9, 'sus4':10,
                    'sus7':11, 'sus9':12, '7(#9)':13, 'min9':14}

# Load configuration
config = nn_config.get_parameters()

# Convert csv files into txt format
csv2txt.CsvToTxt()

#%% Preparing data from bothchroma.csv files
# t2=[]
# rootDir = './McGill_Billboard'
# for dirpath, subdirList, fileList in os.walk(rootDir, topdown=True):
#     for fname in fileList:
#         if fname == 'bothchroma.txt':
#             t2.append(os.path.join(dirpath, fname))
# t2.sort()
t2 = np.load('./data_utils/t2.npy')
print '#files bothchroma.txt: ', len(t2), '\n'

one_hot_csv = np.zeros((len(t2), 1), dtype=object)
frameOn = np.zeros((len(t2), 1), dtype=object)
length = np.zeros((len(t2), 1), dtype=int)

for i in range(len(t2)):
    [frameOn_, one_hot_csv[i,0]] = openWithoutFirstCol(t2[i])
    frameOn[i,0] = frameOn_
    length[i] = len(frameOn_)
    #one_hot_csv[i,0] = csvOneHotEncode(frameOn_, data_csv)

#%% Preparing data from .lab files
# t=[]
# data_lab = []
# rootDir = './McGill_Billboard_lab'
# for dirpath, subdirList, fileList in os.walk(rootDir, topdown=True):
#     for fname in fileList:
#         t.append(os.path.join(dirpath, fname))
# t.sort()
t = np.load('./data_utils/t.npy')
print '#files .lab: ', len(t), '\n'

#matrixOfChords = np.zeros((len(t), 1), dtype=object)
one_hot_lab = np.zeros((len(t), 1), dtype=object)
notes = np.zeros((len(t), 1), dtype=object)
chords = np.zeros((len(t), 1), dtype=object)
t_bins = np.zeros((len(t), 1), dtype=object)
counter = np.zeros((1,len(dictionaryNotes)))

for i in range(len(t)):
    with open(t[i]) as f:
        data_lab = f.read().splitlines()
    f.close()

    [notes[i, 0], chords[i,0], one_hot_lab[i, 0], t_bins[i,0]] = \
        labOneHotEncode(frameOn[i, 0], data_lab, i, dictionaryNotes, dictionaryChords, counter)

#%% Adjust and reshape input/target data for the network
[data, target, target_note, target_ch] = adjustDimension(length, notes, chords, one_hot_csv, one_hot_lab,
                                                         dictionaryNotes, dictionaryChords)

nsignal = 240
nsignalv = 120

data_train = data[:-nsignal]  # 650
data_valid = data[-nsignal:-nsignalv]  # 120
data_test = data[-nsignalv:]  # 120

target_train = target[:-nsignal,:]  # 650
target_valid = target[-nsignal:-nsignalv,:]  # 120
target_test = target[-nsignalv:,:]  # 120
t=t[-nsignalv:]

# Decimate target information since training data is also decimate by the network
target_train = target_train[:,::2,:]
target_valid = target_valid[:,::2,:]
target_test = target_test[:,::2,:]


##### Defining the neural network ####
print 'Building neural network architecture...'

nb_filters = config['nb_filters']
f_length = config['f_len']
load = config['load_weight']
filepath = config['wfilepath']
filepath2 = config['wfilepath2']
pooling = config['pooling']

frame_on = np.load('./data_utils/frame_on.npy')

# Extract note information
target_note_test = target_note[-nsignalv:, :]
target_note_test = target_note_test[:,::2]
target_note_valid = target_note[-nsignal:-nsignalv:, :]
target_note_valid = target_note_valid[:,::2]
target_note_train = target_note[:-nsignal, :]
target_note_train = target_note_train[:,::2]

# Extract chord information
target_ch_test = target_ch[-nsignalv:, :]
target_ch_test = target_ch_test[:,::2]
target_ch_valid = target_ch[-nsignal:-nsignalv:, :]
target_ch_valid = target_ch_valid[:,::2]
target_ch_train = target_ch[:-nsignal, :]
target_ch_train = target_ch_train[:,::2]

m_note = complex_model(nb_filters, pooling, f_length, data_train, target_train[:,:,:18], loading=load, path=filepath)
m_ch = complex_model(nb_filters, pooling, f_length, data_train, target_train[:,:,18:], loading=load, path=filepath2)


if load == 'False':

   #####    NOTES   #####
   ######################
    m_note.summary()
    print "Training the CNN network with note info..."
    best_acc = 0
    init = time()
    for _ in range(config['nb_epoch']):
        hist_n = m_note.fit(
            data_train,
            target_train[:, :, :18],
            validation_data=(data_valid, target_valid[:, :, :18]),
            nb_epoch=1,
            batch_size=config['batch_size'],
            verbose=0
        )

        ### TRAINING ###
        prob_prediction = m_note.predict(x=data_train, batch_size=16, verbose=0)
        frame_on = frame_on[:prob_prediction.shape[1]]
        # Calculating confusion matrix
        confusion_note = np.empty((target_note_train.shape[0], 18, 18), 'int')
        for tt in range(target_note_train.shape[0]):
            note_max = np.argmax(prob_prediction[tt, :, :], axis=1)
            confusion_note[tt, :, :] = confusion_matrix(target_note_train[tt, :], note_max, labels=range(0, 18))

        # Calculate overall accuracy using these confusion matrix
        total = prob_prediction.shape[1]
        note_acc = np.zeros((confusion_note.shape[0], 1), dtype=float)
        for matrix in range(confusion_note.shape[0]):
            note_acc[matrix] = int(confusion_note[matrix].trace()) / total
        print("Accuracy in train (notes): %.4f " % np.mean(note_acc))


        ### VALIDATION ###
        prob_prediction = m_note.predict(x=data_valid, batch_size=16, verbose=0)
        frame_on = frame_on[:prob_prediction.shape[1]]

        # Calculating confusion matrix
        confusion_note = np.empty((target_note_valid.shape[0], 18, 18), 'int')
        for tt in range(target_note_valid.shape[0]):
            note_max = np.argmax(prob_prediction[tt, :, :], axis=1)
            confusion_note[tt, :, :] = confusion_matrix(target_note_valid[tt, :], note_max, labels=range(0, 18))

        # Calculate overall accuracy using these confusion matrix
        total = prob_prediction.shape[1]
        note_acc = np.zeros((confusion_note.shape[0], 1), dtype=float)
        for matrix in range(confusion_note.shape[0]):
            note_acc[matrix] = int(confusion_note[matrix].trace()) / total
        print("Accuracy in validation (notes): %.4f " % np.mean(note_acc))

        if np.mean(note_acc) > best_acc:
            best_acc = np.mean(note_acc)
            m_note.save_weights(filepath)
            print(">> Weights saved!")

    endt = time()
    elapsed_time = endt - init
    print("Elapsed time for training: %.10f seconds." % elapsed_time)

### TEST ###
m_note.load_weights(filepath)
prob_prediction = m_note.predict(x=data_test, batch_size=16, verbose=0)

frame_on = frame_on[:prob_prediction.shape[1]]

# Calculating confusion matrix
confusion_note = np.empty((target_note_test.shape[0], 18, 18), 'int')
note_test=[]
for tt in range(target_note_test.shape[0]):
    note_max = np.argmax(prob_prediction[tt, :, :], axis=1)
    note_test.append(note_max)
    confusion_note[tt, :, :] = confusion_matrix(target_note_test[tt, :], note_max, labels=range(0, 18))

# Calculate overall accuracy using these confusion matrix
total = prob_prediction.shape[1]
note_acc = np.zeros((confusion_note.shape[0], 1), dtype=float)
for matrix in range(confusion_note.shape[0]):
    note_acc[matrix] = int(confusion_note[matrix].trace()) / total
print("Accuracy in test (notes): %.4f " % np.mean(note_acc))

note_test = np.load("note_test.npy")

if load == 'False':

    #####    CHORDS   #####
    #######################
    print "Training the CNN network with chord info..."
    m_ch.summary()

    best_acc = 0
    init = time()
    for _ in range(config['nb_epoch']):
        hist_c = m_ch.fit(
            data_train,
            target_train[:, :, 18:],
            validation_data=(data_valid, target_valid[:, :, 18:]),
            nb_epoch=1,
            batch_size=config['batch_size'],
            verbose=0
        )

        ### TRAINING ###
        prob_prediction = m_ch.predict(x=data_train, batch_size=16, verbose=0)
        frame_on = frame_on[:prob_prediction.shape[1]]

        # Calculating confusion matrix
        confusion_chord = np.empty((target_ch_train.shape[0], 15, 15), 'int')
        for tt in range(target_ch_train.shape[0]):
            chord_max = np.argmax(prob_prediction[tt, :, :], axis=1)
            confusion_chord[tt,:,:] = confusion_matrix(target_ch_train[tt, :], chord_max, labels=range(0, 15))
            # path = './results/' + str(t[tt].split('/')[-2]) + '.txt'
            # save_to_file(note_max, note_max, frame_on, path=path)

        # Calculate overall accuracy using these confusion matrix
        total = prob_prediction.shape[1]
        chord_acc = np.zeros((confusion_chord.shape[0], 1), dtype=float)
        for matrix in range(confusion_chord.shape[0]):
            chord_acc[matrix] = int(confusion_chord[matrix].trace()) / total
        print("Accuracy in train (chord): %.4f " % np.mean(chord_acc))


        ### VALIDATION ###
        prob_prediction = m_ch.predict(x=data_valid, batch_size=16, verbose=0)
        frame_on = frame_on[:prob_prediction.shape[1]]

        # Calculating confusion matrix
        confusion_chord = np.empty((target_ch_valid.shape[0], 15, 15), 'int')
        for tt in range(target_ch_valid.shape[0]):
            chord_max = np.argmax(prob_prediction[tt, :, :], axis=1)
            confusion_chord[tt, :, :] = confusion_matrix(target_ch_valid[tt, :], chord_max, labels=range(0, 15))

        # Calculate overall accuracy using these confusion matrix
        total = prob_prediction.shape[1]
        chord_acc = np.zeros((confusion_chord.shape[0], 1), dtype=float)
        for matrix in range(confusion_chord.shape[0]):
            chord_acc[matrix] = int(confusion_chord[matrix].trace()) / total
        print("Accuracy in validation (chord): %.4f " % np.mean(chord_acc))

        if np.mean(chord_acc) > best_acc:
            best_acc = np.mean(chord_acc)
            m_ch.save_weights(filepath2)
            print(">> Weights saved!")

    endt = time()
    elapsed_time = endt - init
    print("Elapsed time for training: %.10f seconds." % elapsed_time)

### TEST ###
m_ch.load_weights(filepath2)
prob_prediction = m_ch.predict(x=data_test, batch_size=16, verbose=0)

frame_on = frame_on[:prob_prediction.shape[1]]

# Calculating confusion matrix
confusion_chord = np.empty((target_ch_test.shape[0], 15, 15), 'int')
for tt in range(target_ch_test.shape[0]):
    chord_max = np.argmax(prob_prediction[tt, :, :], axis=1)
    confusion_chord[tt, :, :] = confusion_matrix(target_ch_test[tt, :], chord_max, labels=range(0, 15))
    path = './results/' + str(t[tt].split('/')[-2]) + '.txt'
    save_to_file(note_test[tt], chord_max, frame_on, path=path)

# Calculate overall accuracy using these confusion matrix
total = prob_prediction.shape[1]
chord_acc = np.zeros((confusion_chord.shape[0], 1), dtype=float)
for matrix in range(confusion_chord.shape[0]):
    chord_acc[matrix] = int(confusion_chord[matrix].trace()) / total
print("Accuracy in test (chords): %.4f " % np.mean(chord_acc))


embed()


### Plotting results for both NOTES and CHORDS
if config['plot_hist'] == 'True':
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(hist_n.history['acc'])
    plt.plot(hist_n.history['val_acc'])
    plt.title('model accuracy for notes')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure(2)
    plt.plot(hist_n.history['loss'])
    plt.plot(hist_n.history['val_loss'])
    plt.title('model loss for notes')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for accuracy
    plt.figure(3)
    plt.plot(hist_c.history['acc'])
    plt.plot(hist_c.history['val_acc'])
    plt.title('model accuracy for chords')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure(4)
    plt.plot(hist_c.history['loss'])
    plt.plot(hist_c.history['val_loss'])
    plt.title('model loss for chords')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

print 'done \n'


