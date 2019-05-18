import itertools
import argparse
import numpy as np
import pandas as pd
import videoUtils
import np_utils


def to_test_train(avg_fname, all_frames, all_counts, train_ratio=0.8):
    print(all_frames.shape)
    assert len(all_frames) == len(all_counts), 'Frame length should equal counts length'

    nb_classes = all_counts.max() + 1
    X = all_frames

    mean = np.mean(X, axis=0)
    np.save(avg_fname, mean)

    ### Here is to perform random shuffle
    N = len(X) 
    p = np.random.permutation(len(all_counts))
    p = p[0:N]
    Y = np_utils.to_categorical(all_counts, nb_classes)
    assert len(X) == len(Y), 'Len of X (%d) not equal to len of Y (%d)' % (len(X), len(Y))
    X, Y = X[p], Y[p]
    X -= mean

    ### Splitting the data
    def split(arr):
        ind = int(len(arr) * train_ratio)
        return arr[:ind], arr[ind:]

    X_train, X_test = split(X)
    Y_train, Y_test = split(Y)

    return X_train, X_test, Y_train, Y_test

def get_binary(csv_fname, OBJECTS=['car'], limit=None, start=0, WINDOW=30):
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'].isin(OBJECTS)]
    groups = df.set_index('frame')
    counts = list(map(lambda i: i in groups.index, range(start, limit + start)))
    counts = np.array(counts)

    smoothed_counts = np.convolve(np.ones(WINDOW), np.ravel(counts), mode='same') > (WINDOW * 0.7)
    print(np.sum(smoothed_counts != counts), np.sum(smoothed_counts))
    smoothed_counts = smoothed_counts.reshape(len(counts), 1)
    counts = smoothed_counts
    return counts

def get_data_with_split(csv_fname, video_fname, avg_fname,
             num_frames=None, start_frame=0,
             OBJECTS=['car'], resol=(50, 50),
             center=True, dtype='float32', train_ratio=0.8):

    def print_class_numbers(Y, nb_classes):
        classes = np_utils.probas_to_classes(Y)
        for i in range(nb_classes):
            print('class %d: %d' % (i, np.sum(classes == i)))

    print('Parsing %s, extracting %s' % (csv_fname, str(OBJECTS)))
    all_counts = get_binary(csv_fname, limit=num_frames, OBJECTS=OBJECTS, start=start_frame)
   
    print('Retrieving all frames from %s' % video_fname)
    all_frames = videoUtils.get_all_frames(
            len(all_counts), video_fname, scale=resol, start=start_frame)
     
    print('Splitting data into training and test sets')
    X_train, X_test, Y_train, Y_test = to_test_train(avg_fname, all_frames, all_counts, train_ratio = train_ratio)

    nb_classes = all_counts.max() + 1
    print('(train) positive examples: %d, total examples: %d' %
        (np.count_nonzero(np_utils.probas_to_classes(Y_train)),
         len(Y_train)))
    
    print_class_numbers(Y_train, nb_classes)
    print('(test) positive examples: %d, total examples: %d' %
        (np.count_nonzero(np_utils.probas_to_classes(Y_test)),
         len(Y_test)))
    
    print_class_numbers(Y_test, nb_classes)

    print('shape of image: ' + str(all_frames[0].shape))
    print('number of classes: %d' % (nb_classes))

    data = (X_train, Y_train, X_test, Y_test)
    return data, nb_classes

def get_data_for_test(csv_fname, video_fname, avg_fname,
             num_frames=None, start_frame=0,
             OBJECTS=['car'], resol=(50, 50),
             center=True, dtype='float32'):
    # Get the data for avg frame
    avg_frame = np.load(avg_fname)
    all_counts = get_binary(csv_fname, limit=num_frames, OBJECTS=OBJECTS, start=start_frame)
    all_frames = videoUtils.get_all_frames(
            len(all_counts), video_fname, scale=resol, start=start_frame)
    
    nb_classes = all_counts.max() + 1
    X = all_frames
    Y = np_utils.to_categorical(all_counts, nb_classes)
    assert len(X) == len(Y), 'Len of X (%d) not equal to len of Y (%d)' % (len(X), len(Y))
    X -= avg_frame
    # Get all the frame and minus it by the avg frame
    nb_classes = all_counts.max() + 1

    return (X, Y), nb_classes

if __name__ == '__main__':
    
    avg_fname = "/data/dataset/jackson-town-square.npy"
    avg_frame = np.load(avg_fname)
    print(avg_frame.shape)
    pass




