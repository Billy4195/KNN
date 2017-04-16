import numpy as np
import pdb
from os.path import isfile
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import wget
import time

def getWineData():
    if not isfile('winequality-white.csv'):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        wget.download(url)
    data = np.genfromtxt('winequality-white.csv',delimiter=';',skip_header=1)
    return data

def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(11), rotation=45)
    plt.yticks(np.arange(11))
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.show()


def KNN(K,distance,train_set=None,test_set=None,algorithm='brute'):
    print("K = %d, distance = %s, algorithm = %s" % (K,distance,algorithm))
    start_time = time.time()
    if algorithm == 'kd_tree' and distance == 'cosine':
        X,Y = normalize(train_set[:,:11]),train_set[:,11:].ravel()
    else:
        X,Y = train_set[:,:11],train_set[:,11:].ravel()

    nei = KNeighborsClassifier(K,algorithm='brute',metric=distance)
    nei.fit(X,Y)
    train_finish = time.time()
    if algorithm == 'kd_tree' and distance == 'cosine':
        predict = nei.predict(normalize(test_set[:,:11]))
    else:
        predict = nei.predict(test_set[:,:11])
    query_finish = time.time()

    cm = metrics.confusion_matrix(test_set[:,11:].ravel() , predict,np.arange(11))
    print(cm)
#    print("Training time: %f sec" % (train_finish-start_time))
    print("Query time: %f sec" % (query_finish-train_finish))
    accuracy =  nei.score(test_set[:,:11],test_set[:,11:])
    return accuracy, query_finish-train_finish
    #plot_confusion_matrix(cm)


def main():
    data = getWineData()
    for K in range(10,21):
        kf = KFold(n_splits=K)
        for train_index, test_index in kf.split(data):
            train_set = data[train_index]
            test_set = data[test_index]
            print("Train len: %d, test len: %d" % (len(train_set),len(test_set)))
            for algorithm in ['kd_tree','brute']:
                for distance in ['manhattan','cosine','euclidean']:
                    max_acc = 0
                    max_K = 1
                    for i in range(20):
                        accuracy,query_time = KNN(i+1,distance,train_set,test_set,algorithm)
#                        print("Accuracy : %f %%" % (accuracy*100) )
                        if accuracy > max_acc:
                            max_K = i+1
                            max_acc = accuracy
                    with open("result.csv","a") as fp:
                        if algorithm == 'brute':
                            print("%d,%s,%s,%d,%f %%" % (K,algorithm,distance,max_K,max_acc*100), file=fp)

if __name__ == "__main__":
    main()
