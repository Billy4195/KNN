import numpy as np
import pdb
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

def KNN(K,distance=None):
    print("K = %d, distance = %s" % (K,distance))
    data = np.genfromtxt('winequality-white.csv',delimiter=',',skip_header=1)
    X,Y = data[:,:11],data[:,11:].ravel()
    nei = KNeighborsClassifier(K,algorithm='brute',metric=distance)
    nei.fit(X,Y)
    count = 0
    T_count=0
    for idx in range(len(X)):
        count += 1
        if nei.predict(X[idx].reshape((1,-1))) == Y[idx]:
            T_count += 1

    print(T_count/count)

def main():
    for distance in ['manhattan','cosine','euclidean']:
        for i in range(20):
            KNN(i+1,distance)

if __name__ == "__main__":
    main()
