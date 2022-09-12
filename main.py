'''
@File    :   multi_processing.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/9/12 1:13   yifan Jiang      1.0         None
'''

import math
import time
import random
import numpy as np
import pandas as pd
from dotmap import DotMap
from utils import cart2sph, pol2cart, makePath
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv3D,ConvLSTM2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout,AveragePooling3D,MaxPooling3D
from keras.models import Sequential
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from keras.utils import np_utils
from scipy.io import loadmat
import keras
import os
import keras.backend as k
from importlib import reload
np.set_printoptions(suppress=True)
import tensorflow as tf



from scipy.io import loadmat, savemat

def get_logger(name, log_path):
    import logging
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, args):
    locs = loadmat('locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            #KUL_single_single3,contype=no,name=s1,len(arg.ConType)=1
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2:] #KUL,DTU
            # eeg_data = data_pf.iloc[64:, :] #PKU

            data.append(eeg_data)

    data = pd.concat(data, axis=0, ignore_index=True)


    return data

# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split(data, args):
    random.seed(args.random_seed)
    # init
    test_percent = args.test_percent
    window_lap = args.window_length * (1 - args.overlap)#overlap=0.8
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)#overlap_distance=4

    train_set = []
    test_set = []

    for l in range(len(args.ConType)):
        # label = pd.read_csv(args.data_document_path + "/csv/" + args.name + ".csv")#DTU,PKU
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + "No"+".csv")#KUL

        # split trial
        for k in range(args.trail_number):
            # the number of windows in a trial
            window_number = math.floor(
                (args.cell_number - args.window_length) / window_lap) + 1

            test_window_length = math.floor(
                (args.cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1


            test_window_left = random.randint(0, window_number - test_window_length)
            test_window_right = test_window_left + test_window_length - 1

            target = label.iloc[k, args.label_col]

            # split window
            for i in range(window_number):
                left = math.floor(k * args.cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # train set or test set
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, args.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, args.subject_number]))

    # concat
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set

def to_alpha0(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data0 = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[args.point0_low:args.point0_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0/ args.window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha1(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data1 = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data1 = np.abs(window_data1)#1
        window_data1 = np.sum(np.power(window_data1[args.point1_low:args.point1_high, :], 2), axis=0)
        window_data1 = np.log2(window_data1/ args.window_length)
        alpha_data.append(window_data1)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha2(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data2 = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data2 = np.abs(window_data2)/ args.window_length
        window_data2 = np.sum(np.power(window_data2[args.point2_low:args.point2_high, :], 2), axis=0)
        window_data2 = np.log2(window_data2/ args.window_length)
        alpha_data.append(window_data2)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha3(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data3 = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data3 = np.abs(window_data3)
        window_data3 = np.sum(np.power(window_data3[args.point3_low:args.point3_high, :], 2), axis=0)
        window_data3 = np.log2(window_data3 / args.window_length)
        alpha_data.append(window_data3)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha4(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data4 = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data4 = np.abs(window_data4)
        window_data4 = np.sum(np.power(window_data4[args.point4_low:args.point4_high, :], 2), axis=0)
        window_data4 = np.log2(window_data4 / args.window_length)
        alpha_data.append(window_data4)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def main(name="S9", data_document_path="E:\SSF-CNN-master\KUL_single_single3"):
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * 1)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = time.time()
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0

    args.delta_low = 1
    args.delta_high = 3
    args.theta_low = 4
    args.theta_high = 7
    args.alpha_low = 8
    args.alpha_high = 13
    args.beta_low = 14
    args.beta_high = 30
    args.gamma_low = 31
    args.gamma_high = 50
    args.log_path = "./result"


    args.frequency_resolution = args.fs / args.window_length
    args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
    args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
    args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
    args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
    args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
    args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
    args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
    args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    logger = get_logger(args.name, args.log_path)

    # load data 和 label
    data = read_prepared_data(args)


    # split window、testset
    data, train_window, test_window = window_split(data, args)


    train_label = train_window[:, args.window_metadata.target]
    test_label = test_window[:, args.window_metadata.target]

    # fft
    train_data0 = to_alpha0(data, train_window, args)
    test_data0 = to_alpha0(data, test_window, args)
    train_data1 = to_alpha1(data, train_window, args)
    test_data1 = to_alpha1(data, test_window, args)
    train_data2 = to_alpha2(data, train_window, args)
    test_data2 = to_alpha2(data, test_window, args)
    train_data3 = to_alpha3(data, train_window, args)
    test_data3 = to_alpha3(data, test_window, args)
    train_data4 = to_alpha4(data, train_window, args)
    test_data4 = to_alpha4(data, test_window, args)

    #tf.split()

    train_data0 = gen_images(train_data0, args)
    test_data0 = gen_images(test_data0, args)
    train_data1 = gen_images(train_data1, args)
    test_data1 = gen_images(test_data1, args)
    train_data2 = gen_images(train_data2, args)
    test_data2 = gen_images(test_data2, args)
    train_data3 = gen_images(train_data3, args)
    test_data3 = gen_images(test_data3, args)
    train_data4 = gen_images(train_data4, args)
    test_data4 = gen_images(test_data4, args)

    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
    input_test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)

    input_train_data=np.expand_dims(input_train_data,axis=-1)
    input_test_data=np.expand_dims(input_test_data,axis=-1)

    train_label = np_utils.to_categorical(train_label - 1, 2)
    test_label = np_utils.to_categorical(test_label - 1, 2)


    np.random.seed(200)
    np.random.shuffle(input_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(input_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)


    cnn_convlstm_inputs=keras.Input(input_train_data.shape[1:])
    x=Conv3D(32, (3, 3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(0.01),
              data_format="channels_last")(cnn_convlstm_inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=ConvLSTM2D(32,(3,3),padding='same',kernel_regularizer=keras.regularizers.l2(0.01),
                         data_format="channels_last",dropout=0.3)(x)
    x=Flatten()(x)
    x=Dense(512)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(32)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dense(2)(x)
    lianhe_outputs=Activation('softmax')(x)
    model = keras.Model(cnn_convlstm_inputs, lianhe_outputs, name='lianhe_model')
    model.summary()
    opt = keras.optimizers.RMSprop(lr=0.0003, decay=3e-4)#RMSprop优化器，
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(input_train_data, train_label, batch_size=args.batch_size, epochs=args.max_epoch, validation_split=args.vali_percent, verbose=2,shuffle=True)

    loss, accuracy = model.evaluate(input_test_data, test_label)
    dense=tf.keras.models.Model(inputs=model.input, outputs=model.layers[13].output)
    dense_out=model.predict(input_train_data)
    print('This is dense_out', dense_out)
    print(loss, accuracy)
    logger.info(loss)
    logger.info(accuracy)
    logger.info(dense_out)


if __name__ == "__main__":
    main()
