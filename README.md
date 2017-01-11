# chord_estimation
In this project I estimate a sequence of chords from audio music recordings using a convolutional neural networks as classifier.

>>> NOTE: Project based on Billaboard dataset (McGill University). Download "chordino" training dataset and "LAB files" with ground truth annotations from the website http://ddmal.music.mcgill.ca/research/billboard 

The harmonic structure of different audio tracks from the Billaboard dataset (McGill University) were extracted. This task required the estimation of a sequence of chords which was as precise as possible, which included the full characterisation of chords – root, quality, and bass note – as well as their chronological order, including specific onset times and durations. To evaluate the quality of the automatic transcriptions, each one was compared to ground truth previously created by a human annotator.

Before training the network and estimate the chords of the soundtracks, an intensive data preprocessing is needed in order to extract the useful data and reshape it into a suitable format to feed the network. The variables stored in ./data_utils help to speed up this process. 

A full chord is composed by two parts: the root note and the scale chord separated by a colon (i.e. A:maj). Thus, I define two dictionaries with each note and chord vocabulary. Following this idea, I train the same network architecture with notes' information and with chords separetely. This is, in my implementation there is no relationship between notes and chords and both throw its own results. Talking about classification task, each word on each dictionary – e.g., ‘D#’, ‘maj’ – is counted as a different class. So we count to 18 classes for notes and 15 for chords (for further information see definition in main.py).

Two different convolutional neural networks are designed and tested. They can be found in utils_.py as "baseline_model" and "complex_model". The network outline and its performance can be found in ./data_utils/results.pdf file. (Run with GPU)

