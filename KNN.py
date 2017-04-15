import numpy as np
import pdb
from os.path import isfile
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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


def KNN(K,distance,train_set=None,test_set=None,algorithm='brute'):
    print("K = %d, distance = %s, algorithm = %s" % (K,distance,algorithm))
    if algorithm == 'kd_tree' and distance == 'cosine':
        X,Y = normalize(train_set[:,:11]),train_set[:,11:].ravel()
    else:
        X,Y = train_set[:,:11],train_set[:,11:].ravel()

    nei = KNeighborsClassifier(K,algorithm='brute',metric=distance)
    nei.fit(X,Y)

    if algorithm == 'kd_tree' and distance == 'cosine':
        predict = nei.predict(normalize(test_set[:,:11]))
    else:
        predict = nei.predict(test_set[:,:11])

    cm = metrics.confusion_matrix(test_set[:,11:].ravel() , predict,np.arange(11))
#    print(cm)
    print(nei.score(test_set[:,:11],test_set[:,11:]))
    #plot_confusion_matrix(cm)


def main():
    data = getWineData()
    train_set, test_set = train_test_split(data, test_size=0.05)
    for algorithm in ['kd_tree','brute']:
        for distance in ['manhattan','cosine','euclidean']:
            for i in range(20):
                    KNN(i+1,distance,train_set,test_set,algorithm)

if __name__ == "__main__":
    main()
