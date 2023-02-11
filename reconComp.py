#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pcplot import *
from matplotlib.offsetbox import AnchoredText


def reconEval(label,Y_tilda,Y_til_recon,title):

    ###### visualize Y_tilda & Y_til_recon ###########
    Y_tilda_tran = pcplot(label, Y_tilda, title)
    plt.savefig('./results/'+title+'1.png')
    Y_til_recon_tran = pcplot(label, Y_til_recon, title)
    plt.savefig('./results/'+title+'2.png')

    d_vec = np.sqrt(np.sum((Y_tilda_tran - Y_til_recon_tran)**2, axis=1))

    ###### quantitize the recon error #########

    reconErrArray = np.sqrt(np.mean((Y_tilda - Y_til_recon)**2,axis=1)/np.mean(Y_tilda**2,axis=1))

    return reconErrArray, d_vec, Y_til_recon_tran

def compPlot(atta1,atta2,atta3,yName):

    fig, ax = plt.subplots()
    decimal = 4

    ########### plot ##########

    plot_data = pd.DataFrame()
    plot_data[yName] = np.hstack([atta1,atta2,atta3])

    n1 = np.size(atta1,0)
    n2 = np.size(atta2,0)
    n3 = np.size(atta3,0)

    method = ['MCU' for i in range(n1)] + ['MVU' for i in range(n2)]+['PCA' for i in range(n3)]
    plot_data['method'] = method

    medians = plot_data.groupby(['method'])[yName].median()

    box_plot = sns.boxplot(x="method", y=yName, data=plot_data, color="skyblue")#, showfliers=False) # palette="PRGn")
    vertical_offset = plot_data[yName].median() * 0.05

    for xtick in box_plot.get_xticks():
        box_plot.text(xtick,medians[xtick] + vertical_offset,np.round(medians[xtick],decimal),
                horizontalalignment='center',size='x-small',color='black',weight='semibold')

    at = AnchoredText("IQR: "+str(np.round(np.percentile(atta1,75)-np.percentile(atta1,25),decimal)),
                  prop=dict(size=8), frameon=True,
                  loc='upper left',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    at = AnchoredText("IQR: "+str(np.round(np.percentile(atta2,75)-np.percentile(atta2,25),decimal)),
                  prop=dict(size=8), frameon=True,
                  loc='upper center',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    at = AnchoredText("IQR: "+str(np.round(np.percentile(atta3,75)-np.percentile(atta3,25),decimal)),
                  prop=dict(size=8), frameon=True,
                  loc='upper right',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
