# Imports
import numpy as np
import pickle
import random

dir = 'CIFAR/cifar-10-batches-py/'
datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
testdatafile = ['test_batch']

# Rotate and adjust image :  Rotate image by 90 degrees clockwises
def rotate(data_batch):
    for i in range(len(data_batch)):
        data_batch[i] = np.rot90(data_batch[i])
        data_batch[i] = np.rot90(data_batch[i])
        data_batch[i] = np.rot90(data_batch[i])
    return data_batch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Make CIFAR10 test and train data
def cifar10():

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # for one hot encodinng
    y_train_onehot = np.zeros(shape=(50000,10), dtype=int, order='C')
    y_test_onehot = np.zeros(shape=(10000,10), dtype=int, order='C')

    # training data
    for file in datafiles:
        data = unpickle('CIFAR/cifar-10-batches-py/'+file)

        print('loaded ' + file)
        print('len of ' + file + ' : ' ,len(data[b'data']))

        # merge 5 trainig fils into one
        x_train[len(x_train):] = data[b'data']
        y_train[len(y_train):] = data[b'labels']

        #x_train.extend(data[b'data'])
        #y_train.extend(data[b'labels'])

        print('len of training data ',
         len(x_train))
        print('================')


    # test data
    testdata = unpickle('CIFAR/cifar-10-batches-py/test_batch')
    x_test[len(x_test):] = testdata[b'data']
    y_test[len(y_test):] = testdata[b'labels']

    #change type to np arrray
    x_train = np.asarray(x_train, order='C')
    x_test = np.asarray(x_test, order='C')

    # change shape of test and train images (_ x 32 x 32 x 3)
    x_train = np.reshape(x_train, (50000, 32, 32, 3), order='F')
    x_test = np.reshape(x_test, (10000, 32, 32, 3), order='F')

    # Rotate train and test images
    x_train = rotate(x_train)
    x_test = rotate(x_test)

    # One hot encoding of y_train and y_test
    for i in range(len(y_train)):
        y_train_onehot[i][y_train[i]] = 1

    for i in range(len(y_test)):
        y_test_onehot[i][y_test[i]] = 1

    #print('=============================')
    #print('full data info:')
    #print('x_train shape:', x_train.shape)
    #print('y_train shape:', y_train_onehot.shape)
    #print('x_test shape: ', x_test.shape)
    #print('y_test shape:', y_test_onehot.shape)

    #Image is Channel last formate
    return x_train, y_train_onehot, x_test, y_test_onehot, y_train, y_test


# CIFAR small (Class<=10)
def cifar(totalClass=10, shufflePixels=False):

    # Load Cifar10 data
    x_train, y_train, x_test, y_test, a, b = cifar10()

    if(totalClass == 10 and shufflePixels == False):
        return x_train, y_train, x_test, y_test, a, b

    #No. of training samples per class = 5000
    #No. of test samples per class = 1000

    #Trim Data as per class
    xxtrain = np.zeros((totalClass*5000,32,32,3), dtype='int16')
    yytrain = np.zeros((totalClass*5000,totalClass), dtype='int16')
    xxtest = np.zeros((totalClass*1000,32,32,3), dtype='int16')
    yytest = np.zeros((totalClass*1000,totalClass), dtype='int16')

    #Making Train data
    index = 0
    for i in range(50000):
        if (np.argmax(y_train[i]) < totalClass):
            label = np.argmax(y_train[i])
            image = x_train[i]
            xxtrain[index] = image.copy()
            yytrain[index, label] = 1
            index = index + 1

    #Making  Test data
    index = 0
    for i in range(10000):
        if (np.argmax(y_test[i]) < totalClass):
            label = np.argmax(y_test[i])
            image = x_test[i]
            xxtest[index] = image.copy()
            yytest[index, label] = 1
            index = index + 1

    #Simple Y train (not hot encoded)
    a = np.zeros((len(yytrain)))
    b = np.zeros((len(yytest)))

    for i in range(len(yytrain)):
        a[i] = np.argmax(yytrain[i])

    for i in range(len(yytest)):
        b[i] = np.argmax(yytest[i])

    if (shufflePixels == False):
        return xxtrain, yytrain, xxtest, yytest, a, b

    if (shufflePixels == True):
        xxtrain_, yytrain, xxtest_, yytest = shuffledata(xxtrain, yytrain, xxtest, yytest)
        return xxtrain_, yytrain, xxtest_, yytest, a, b



def shuffledata(xtrain, ytrain, xtest, ytest, shufflePixels=True, shuffleLabels=False):

    if shufflePixels == True:
        print(xtrain.shape)
        no_of_train_samples, height, width, channel = xtrain.shape
        xtrain = np.reshape(xtrain, (no_of_train_samples, 3072)) # 32 * 32 * 3 = 3072
        no_of_test_samples, height, width, channel = xtest.shape
        xtest = np.reshape(xtest, (no_of_test_samples, 3072)) # 32 * 32 * 3 = 3072

        # Shuffle train images
        for i in range(no_of_train_samples):
            random.shuffle(xtrain[i])

        #Shuffle test images
        for i in range(no_of_test_samples):
            random.shuffle(xtest[i])

        # Reshape
        xtrain = np.reshape(xtrain, (no_of_train_samples, 32, 32, 3))
        xtest = np.reshape(xtest, (no_of_test_samples, 32, 32, 3))

    if shuffleLabels == True:
        for i in range(no_of_train_samples):
            random.shuffle(ytrain[i])

        for i in range(no_of_test_samples):
            random.shuffle(ytest[i])

    return xtrain, ytrain, xtest, ytest

"""
def channelFirstToLast(xtrain, xtest):
    #some code <Not required>

"""
