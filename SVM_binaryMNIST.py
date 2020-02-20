

# ----Train a SVM based on 2-class MNIST----
# Feb-2020

from mlxtend.data import loadlocal_mnist
import numpy as np
from matplotlib import pyplot
from keras.datasets.mnist import load_data 
from numpy import expand_dims
from sklearn.svm import SVC


#<<<< load mnist images >>>>
def load_mnist_samples(positive,negative):
    # load mnist dataset
    (trainSet, trainLabel), (testSet, testLabel) = load_data()
    
    #filter negative / positive samples
    train_filter_negative = np.where(trainLabel == negative) 
    test_filter_negative = np.where(testLabel == negative)  
    
    trainSet_negative, trainLabel_negative = trainSet[train_filter_negative], trainLabel[train_filter_negative]
    testSet_negative, testLabel_negative = testSet[test_filter_negative], testLabel[test_filter_negative]
    
    #filter positive samples
    train_filter_positive = np.where(trainLabel == positive) 
    test_filter_positive = np.where(testLabel == positive)  
    
    trainSet_positive, trainLabel_positive = trainSet[train_filter_positive], trainLabel[train_filter_positive]
    testSet_positive, testLabel_positive = testSet[test_filter_positive], testLabel[test_filter_positive]                                     
    # convert to +1 / -1
    if negative == 0:
        trainLabel_negative = -1 + trainLabel_negative    
        testLabel_negative = -1 + testLabel_negative 
    else:
        trainLabel_negative = -1 * (trainLabel_negative / negative)   
        testLabel_negative = -1 * (testLabel_negative / negative)
        
    if positive == 0:        
        trainLabel_positive = +1 + trainLabel_positive 
        testLabel_positive = +1 + testLabel_positive 
    else:
        trainLabel_positive = +1 * (trainLabel_positive / positive)   
        testLabel_positive = +1 * (testLabel_positive / positive)   
        
    # merge
    X = np.concatenate((trainSet_negative, trainSet_positive), axis=0) 
    X_l = np.concatenate((trainLabel_negative, trainLabel_positive), axis=0)
    Y = np.concatenate((testSet_negative, testSet_positive), axis=0) 
    Y_l = np.concatenate((testLabel_negative, testLabel_positive), axis=0)                                     
    # expand to 1x784
    X = np.reshape(X, (-1,28*28))
    X_l = np.reshape(X_l, (-1,1))
    Y = np.reshape(Y, (-1,28*28)) 
    Y_l = np.reshape(Y_l, (-1,1))
    # convert from unsigned ints to floats
    X = X.astype('float32')
    Y = Y.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0
    Y = Y / 255.0

    return X,Y,X_l,Y_l

  



# ------------------ Main loop-----------------------
svm_samples = 300

# positive ~= negative
positive = 0
negative = 6

# Here, I load my data
(X,Y,X_l,Y_l) = load_mnist_samples(positive,negative)
# shuffle training data
combined_X = list(zip(X, X_l))
combined_Y = list(zip(Y, Y_l))
np.random.shuffle(combined_X)
np.random.shuffle(combined_Y)
X[:], X_l[:] = zip(*combined_X)
Y[:], Y_l[:] = zip(*combined_Y)
#
#print(X.shape)
#print(X_l.shape)
#print(Y.shape)
#print(Y_l.shape)

# prepare svm_data
X_svm = X[1:X.shape[0],:]
X_l_svm = X_l[1:X.shape[0]]
#print(X_svm.shape)
#print(X_l_svm.shape)

# train SVM
clf = SVC(gamma='auto')
clf.fit(X_svm,X_l_svm)


# Here, I pass my data with suitable shape to train a SVM
#model_svm = train_svm(X_svm,X_l_svm,epochs=5000,eta=.2)

print("done!")





#cnt = 0
#for ii in range (X_svm.shape[0]):
#    tt= X_svm[ii].reshape((-1,28*28))
#    #print(clf.predict((tt)),X_l_svm[ii])  
#    if (clf.predict((tt))==X_l_svm[ii]):
#        cnt = cnt
#    else:
#        cnt = cnt + 1
#print(X_svm.shape[0],cnt)



