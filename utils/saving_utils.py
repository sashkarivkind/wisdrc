
"""
this file provides manual methods for saving model,
to mitigate problems with saving recurrent convolutional layers in keras
"""

import os
import pickle

def mkdir_if_needed(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def save_model_old(net, path, parameters, checkpoint=True):
    home_folder = path + '{}_saved_models/'.format(this_run_name)
    if not os.path.exists(home_folder):
        os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)

    # save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(this_run_name)
    if not os.path.exists(keras_weights_path):
        os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}'.format(this_run_name))
    # LOADING WITH - load_status = sequential_model.load_weights("ckpt")

"""
legacy version:
"""

def save_model_old(net, path, parameters, checkpoint=True):
    home_folder = path + '{}_saved_models/'.format(this_run_name)
    if not os.path.exists(home_folder):
        os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)

    # Saving using net.save method
    model_save_path = child_folder + '{}_keras_save'.format(this_run_name)
    # os.mkdir(model_save_path)
    # net.save(model_save_path)
    # Saving weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(this_run_name)
    if not os.path.exists(numpy_weights_path):
        os.mkdir(numpy_weights_path)
    all_weights = net.get_weights()
    with open(numpy_weights_path + 'numpy_weights_{}'.format(this_run_name), 'wb') as file_pi:
        pickle.dump(all_weights, file_pi)
    # LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()

    # save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(this_run_name)
    if not os.path.exists(keras_weights_path):
        os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}'.format(this_run_name))
    # LOADING WITH - load_status = sequential_model.load_weights("ckpt")
