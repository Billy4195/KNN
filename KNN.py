import numpy as np
import pdb
from os.path import isfile
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import matplotlib.pyplot as plt
import wget

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


def KNN(K,distance,data,training_data=None,test_data=None):
    print("K = %d, distance = %s" % (K,distance))
    X,Y = data[:,:11],data[:,11:].ravel()
    nei = KNeighborsClassifier(K,algorithm='brute',metric=distance)
    nei.fit(X,Y)
    predict = nei.predict(X)
    cm = metrics.confusion_matrix(Y , predict,np.arange(11))
    print(cm)
    #plot_confusion_matrix(cm)


def main():
    data = getWineData()
    for distance in ['manhattan','cosine','euclidean']:
        for i in range(1):
            KNN(i+1,distance,data)

if __name__ == "__main__":
    main()
