#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li               #
#  USC ID: 6503378943              #
#  USC Email: zongjian@usc.edu     #
#  Submission Date: 7th, Apr 2019  #
####################################

def main():
    # read args
    import argparse
    parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 5 by Zongjian Li.")
    parser.add_argument("-n", "--negtive_train", action="store_true", help="Expand the trainning data set using negtive images.")
    parser.add_argument("-e", "--epochs", type=int, default=15, help="Epochs of trainning.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="Batch size of trainning.")
    parser.add_argument("-o", "--output_filename", type=str, default="trained_model", help="Filename to save trained model.")
    args = parser.parse_args()

    # define
    import keras
    import keras.layers as layers
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=10, activation = 'softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # padding
    import numpy as np
    pad_width = (32 - 28) // 2
    x_train = np.pad(x_train, ((0, ), (pad_width, ), (pad_width, )), "edge")
    x_test = np.pad(x_test, ((0, ), (pad_width, ), (pad_width, )), "edge")

    # expand dim
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # one-hot
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # negtive train
    x_train = np.concatenate((x_train, np.subtract(255, x_train)), axis=0) if args.negtive_train else x_train
    y_train = np.concatenate((y_train, y_train), axis=0) if args.negtive_train else y_train

    # train 
    # NOTE: 
    # Since I do not need to adjust the network, just implement the LeNet-5, so validation data set is not needed. 
    # And the requirement asks me to record the acc both on training and test data sets for every epoch, so I set "validation_data" to test data set.
    from datetime import datetime
    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[tensorboard])

    # evaluate negtive test
    x_test_neg = np.subtract(255, x_test)
    score = model.evaluate(x_test_neg, y_test, verbose=False)
    print("Negtive test loss={0} accuracy={1}".format(*score))

	# save
    model.save(args.output_filename)

if __name__ == "__main__":
    main()