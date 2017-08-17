#Xi Pardee
#opencv package is required to make the greyscale picture

import pandas as pd
import numpy as np
import scipy.spatial
from sklearn.metrics import confusion_matrix
import cv2
# number of cluster
k = 30

# calculatethe entropy for each cluster
def centropy(label):
    size = float(label.size)
    total = 0
    u, c = np.unique(label, return_counts=True)
    for i in range(u.shape[0]):
        total += (c[i]/size)*np.log2(c[i]/size)

    i = np.argmax(c)
    return -total, u[i]

# calculate the mean entropy
def meanEntropy(instanceCount, entropy):
    total = np.sum(instanceCount * entropy)
    return total/3823

# run each iteration
def iteration(dataSet, centroid, mss, entropy, instanceCount, finalLabel):
    distance = scipy.spatial.distance.cdist(dataSet, centroid)
    index = np.argmin(distance, axis=1)
    empty = np.full((64), -1)
    oldCentroid = np.copy(centroid)

    # update the cluster center
    for i in range(k):
        indexsNumber = np.where(index == i)
        instances = dataSet.values[indexsNumber]
        label = dataSet.index.values[indexsNumber]
        instanceCount[i] = instances.shape[0]
        if instances.size != 0:
            centroid[i] = np.mean(instances, axis=0)
            mss[i] = np.sum(np.square(instances - centroid[i]))/instances.shape[0]
            entropy[i], finalLabel[i] = centropy(label)
        else:
            centroid[i] = empty

    # for empty cluster, choose a random center from the data set
    for i in range(k):
        if centroid[i][0] == -1:
            index = np.random.randint(dataSet.shape[0])
            centroid[i] = np.copy(dataSet.values[index])

    equ = np.array_equal(oldCentroid, centroid)
    return equ

# calculate the mean square separation
def msp(seeds):
    sum = 0
    for i in range(k):
        for j in range(i+1, k):
           sum += np.sum(np.square(seeds[i] - seeds[j]))
    return sum/(k*(k-1)*2)

# classify the test data
def classify(seeds, finalLabel, testData):
    distance = scipy.spatial.distance.cdist(testData, seeds)
    cl = np.argmin(distance, axis=1)
    pre = np.zeros(cl.shape[0])
    for i in range(cl.shape[0]):
        pre[i] = finalLabel[cl[i]]
    return pre

if __name__ == "__main__":
    data = pd.read_csv('optdigits.train', header=None, index_col=64)
    fseeds = np.random.randint(0, 17, size=(k, 64))
    ffinalLabel = np.zeros(k)
    famse = 10000000

    # create clusters 5 times and choose the one with lowest average mean square error
    for i in range(5):
        seeds = np.random.randint(0, 17, size=(k, 64))
        mss = np.zeros(k)
        instanceCount = np.zeros(k)
        entropy = np.zeros(k)
        finalLabel = np.zeros(k)
        equal = False
        times = 1
        while (equal == False):
            print ('iteration: ', times )
            equal = iteration(data, seeds, mss, entropy, instanceCount, finalLabel)
            times += 1
        amse = np.mean(mss)
        mspp = msp(seeds)
        me = meanEntropy(instanceCount, entropy)
        print('average mean-square-error; ' + str(amse))
        print('mean square separation: ' + str(mspp))
        print('mean entropy: ' + str(me))
        print("")
        if amse < famse:
            famse = amse
            fseeds = seeds
            ffinalLabel = finalLabel

    print(ffinalLabel)
    # read in the test data and use the clusters to classify them
    testData = pd.read_csv('optdigits.test', header=None, index_col=64)
    tru = testData.index.values
    pre = classify(fseeds, ffinalLabel, testData)
    matrix = confusion_matrix(tru, pre)

    print (matrix)
    accuracy = float(np.sum(tru == pre))/tru.shape[0]
    print('accuracy rate of test data: ' + str(accuracy))

    # create the grey scale picture
    for i in range(k):
        img = fseeds[i].reshape(8,8)
        img = img * 255 / 16
        img = img.astype(np.uint8)
        height, width = img.shape[:2]
        img = cv2.resize(img, (5 * height, 5 * width), interpolation=cv2.INTER_NEAREST)
        frame = 'frame' + str(i)
        cv2.imshow(frame, img)
        name = str(i)+'.jpg'
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(name, img)
            cv2.destroyAllWindows()
