#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

def pcplot(label, Y_tilda, title, Xlabel=None):

    Y_tilda_dim2 = Y_tilda[:,0:2]
    #standardization
    std = np.std(Y_tilda_dim2, axis=0)
    Y_tilda = Y_tilda_dim2 / np.mean(std)

    fig = plt.figure()
    ax = fig.gca()
    for l in np.unique(label):
        ax.scatter(Y_tilda[label == l, 0], Y_tilda[label == l, 1],
                   color=plt.cm.jet(np.float(l) / np.max(label + 1)),
                   s=20, edgecolor='k')

    if Xlabel is not None:
        xs = Y_tilda[:,0]
        ys = Y_tilda[:,1]

        label_list = [np.array2string(Xlabel[i,:]) for i in range(Xlabel.shape[0])]

        for x, y, label in zip(xs, ys, label_list):
            ax.text(x, y, label)

    # Tweaking display region and labels
    #ax.set_xlim(-3.5, 3.5)
    #ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    plt.title(title)

    return Y_tilda


def pcplot2(label, Y_tilda, title, Xlabel=None):

    Y_tilda_dim3 = Y_tilda[:,0:3]
    #standardization
    std = np.std(Y_tilda_dim3, axis=0)
    Y_tilda = Y_tilda_dim3 / np.mean(std)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.scatter(Y_tilda[label == l, 0], Y_tilda[label == l, 1], Y_tilda[label == l, 2],
                   color=plt.cm.jet(float(l) / np.max(label + 1)),
                   s=20, edgecolor='k')

    if Xlabel is not None:

        label_list = [np.array2string(Xlabel[i,:]) for i in range(Xlabel.shape[0])]

        xs = Y_tilda[:,0]
        ys = Y_tilda[:,1]
        zs = Y_tilda[:,2]

        for x, y, z, label in zip(xs, ys, zs, label_list):
            ax.text(x, y, z, label)

    # Tweaking display region and labels
    #ax.set_xlim(-3.5, 3.5)
    #ax.set_ylim(-3.5, 3.5)
    #ax.set_zlim(-1, 1)

    #ax.set_xlabel('X axis')
    #ax.set_ylabel('Y axis')
    #ax.set_zlabel('Z axis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.title(title)
