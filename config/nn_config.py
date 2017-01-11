
def get_parameters():
    nn_params = {}

    nn_params['nb_epoch'] = 20
    nn_params['batch_size'] = 64
    nn_params['nb_filters'] = 16
    nn_params['pooling'] = 2
    nn_params['f_len'] = 2


    nn_params['load_weight'] = 'False'
    nn_params['save_weight'] = 'False'
    nn_params['wfilepath'] = './weights/weights-v3-notes.hdf5'
    nn_params['wfilepath2'] = './weights/weights-v3-chords.hdf5'

    nn_params['file_matrix'] = './confusion_matrix_notes.txt'
    nn_params['plot_hist'] = 'False'

    return nn_params
