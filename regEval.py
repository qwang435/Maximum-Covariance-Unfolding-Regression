import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import optimize
from sklearn.decomposition import PCA

class regEvaluator():

    def __init__(self,Xtest,Ytest,pca0):
        self.Xtest = Xtest
        self.Ytest = Ytest

        self.n_test = np.size(Xtest,0)
        self.n_ctr = np.size(Xtest,1)

        self.Ytest0 = pca0.transform(Ytest)

    def regEval(self,X,Xmean,Y_tilda,Y0,k,lamb):

        ####### learn regression parameter ############

        beta = np.linalg.inv(X.T@X+lamb*np.eye(np.size(X,1)))@X.T@Y_tilda
        Y_til_recon = X@beta

        ######### process optimization #########

        nbrs = NearestNeighbors(n_neighbors=k).fit(Y0)
        Xpred = np.zeros((self.n_test,self.n_ctr))

        for i in range(self.n_test):
            if i % 20 == 0:
                print(i)
            nominal = self.Ytest0[i,:][np.newaxis,:]
            distances, indices = nbrs.kneighbors(nominal)
            bounds = list(zip(np.min(X, axis=0),np.max(X, axis=0)))

            obj= lambda xnew: np.linalg.norm(np.linalg.norm(xnew.reshape(1,self.n_ctr)@beta
                                                            - Y_tilda[indices[0],:],
                                                            axis=1)
                                             - distances[0])

            Xpred[i,:] = Xmean + optimize.dual_annealing(obj,bounds).x

        XpredErrArray = np.linalg.norm(Xpred - self.Xtest,axis=1)

        return Y_til_recon, XpredErrArray, Xpred

    def regEvalPCA(self,X,Xmean,Y_tilda,lamb, pca):

        YtestPCA = pca.transform(self.Ytest)

        ####### learn regression parameter ############

        beta = np.linalg.inv(X.T@X+lamb*np.eye(np.size(X,1)))@X.T@Y_tilda
        Y_til_recon = X@beta

        ######### process optimization #########

        Xpred = np.zeros((self.n_test,self.n_ctr))

        for i in range(self.n_test):
            if i % 20 == 0:
                print(i)

            bounds = list(zip(np.min(X, axis=0),np.max(X, axis=0)))
            obj= lambda xnew: np.linalg.norm(xnew@beta-YtestPCA[i,:])
            Xpred[i,:] = Xmean + optimize.dual_annealing(obj,bounds).x

        XpredErrArray = np.linalg.norm(Xpred - self.Xtest,axis=1)

        return Y_til_recon, XpredErrArray, Xpred
