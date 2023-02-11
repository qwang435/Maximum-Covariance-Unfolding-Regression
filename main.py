#!/usr/bin/env python
import numpy as np
from scipy.io import loadmat
import sys
from sklearn.decomposition import PCA
from MXU import *
from reconComp import *
from regEval import *
from sklearn.cluster import AgglomerativeClustering
from pcplot import *

###### read and preprocess ############

X = np.array(loadmat('./data/XY.mat')['X'])
Y = np.array(loadmat('./data/XY.mat')['Y'])
Xlabel = np.around(X,decimals=2)

n1 = np.size(X,0)
n2 = np.size(X,1)

Xmean = np.mean(X,axis=0)
Xstd = np.std(X, axis=0)
X = (X - Xmean) #/ Xstd

#Ymean = np.mean(Y,axis=0)
#Y_center = Y - Ymean
#Ystd = np.std(Y, axis=0)
#Y = Y_center/np.mean(Ystd)

Xtest = np.array(loadmat('./data/XYtest.mat')['Xtest'])
Ytest = np.array(loadmat('./data/XYtest.mat')['Ytest'])

pca0 = PCA(n_components=100)
Y0 = pca0.fit_transform(Y)
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Y0)
label = ward.labels_
pcplot2(label,Y0,"manifold shape")
plt.show()

################ main ##################

d = 2
assert d <= n1 #and d <= np.size(Xtest,0)

pca = PCA(n_components=d)
Y_tilda3 = pca.fit_transform(Y)

regEvaluator = regEvaluator(Xtest,Ytest,pca0)

####### hyperparameters ########
switch = 'plain' #'plain','kernel','kernel2'

if switch == 'plain':
    lamb = 1e-10
    k1 = 4
    k2 = 4

    Y_tilda1 = MCU(X,Y0,k1)[:,:d]
    Y_tilda2 = MVU(Y0,k2)[:,:d]

    Y_til1_recon, XpredErrArray1, Xpred1 = regEvaluator.regEval(X,Xmean,Y_tilda1,Y0,k1,lamb)
    Y_til2_recon, XpredErrArray2, Xpred2 = regEvaluator.regEval(X,Xmean,Y_tilda2,Y0,k2,lamb)
    Y_til3_recon, XpredErrArray3, Xpred3 = regEvaluator.regEvalPCA(X,Xmean,Y_tilda3,lamb,pca)

else:
    print('argument incorrect!')
    sys.exit(0)


######## regression comparison ##########

reconErrArray1, d_vec1, Y_til1_recon_tran = reconEval(label, Y_tilda1,Y_til1_recon,"MCU")
reconErrArray2, d_vec2, Y_til2_recon_tran = reconEval(label, Y_tilda2,Y_til2_recon,"MVU")
reconErrArray3, d_vec3, Y_til3_recon_tran = reconEval(label, Y_tilda3,Y_til3_recon,"PCA")

#color_min = np.minimum(np.min(d_vec1),np.min(d_vec2),np.min(d_vec3))
color_max = np.max([np.max(d_vec1),np.max(d_vec2),np.max(d_vec3)])

plt.figure()
plt.subplot(1,3,1)
plt.scatter(x=Y_til1_recon_tran[:,0], y=Y_til1_recon_tran[:,1], c=d_vec1, vmin=0, vmax=color_max,cmap='jet', s=20)#, edgecolor='k')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('MCU')
plt.subplot(1,3,2)
plt.scatter(x=Y_til2_recon_tran[:,0], y=Y_til2_recon_tran[:,1], c=d_vec2, vmin=0, vmax=color_max,cmap='jet', s=20)#, edgecolor='k')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('MVU')
plt.subplot(1,3,3)
plt.scatter(x=Y_til3_recon_tran[:,0], y=Y_til3_recon_tran[:,1], c=d_vec3, vmin=0, vmax=color_max,cmap='jet', s=20)#, edgecolor='k')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('PCA')
plt.colorbar()
plt.savefig('./results/regcompdistance.png')

#compPlot(XpredErrArray1,XpredErrArray2,XpredErrArray3,'Nominal X optimization error')
#plt.savefig('./plots/X opt.png')
print('nominal x:', Xtest)
print('optimal x from MCU:', Xpred1)
print('optimal x from MVU:', Xpred2)
print('optimal x from PCA:', Xpred3)

compPlot(reconErrArray1,reconErrArray2,reconErrArray3,'Relative Y_tilda reconstruction error')
plt.savefig('./results/regcomp.png')

plt.show()
