
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['image.cmap'] = 'gist_ncar'  # change default colormap

import matplotlib.patches as patches
import numpy as np


def unique_cm_dict_from_list(items, basecmap = 'gist_rainbow'):
    # From Sacrum
    cm = plt.get_cmap(basecmap)
    cnorm = mpl.colors.Normalize(vmin=0, vmax=len(items))
    scalarmap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cm)
    return dict(zip(items, [scalarmap.to_rgba(i) for i in range(len(items))]))

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlabscsv", help="XLabs Output CSV for Visualisation")
    parser.add_argument("-c", "--calicut", action='store', default=None, help="Estimated index of the end of Calibration")
    parser.add_argument("-x", "--xlim", action='store', default=1920, help="Restrict view to given x limit")
    parser.add_argument("-y", "--ylim", action='store', default=1080, help="Restrict view to given y limit")
    parser.add_argument("--xslim", action='store', default=1, help="Restrict view to given x limit for the scaled view")
    parser.add_argument("--yslim", action='store', default=1, help="Restrict view to given y limit for the scaled view")
    parser.add_argument("-Y", "--invert_y", action='store_true', default=False, help="Invert y-axis")
    parser.add_argument("-C", "--confidence_factor", action='store', default=1.0, help='Limit output to given percentage confidence (i.e. 0.8 drops 20%% worst)')

    return parser

def go(xlabscsv, calicut=None, invert_y=False, xlim=None, ylim=None, xslim=None, yslim=None, confidence_factor=None, **kwargs):
    df = pd.read_csv(xlabscsv,
                     names=["Xr", "Yr", "Xp", "Xs", "Yp", "Ys", "C", 3, ],
                     index_col=False)
    df.reset_index(inplace=True)
    if confidence_factor is not None:
        df = df[df.C < df.C.quantile(confidence_factor)]

    if calicut is not None:
        fig, sup_axes = plt.subplots(2,3, figsize = (16,10))
    else:
        fig, sup_axes = plt.subplots(1,3, figsize = (16,10))



    for i,(x,y) in enumerate(zip(['Xr','Xp', 'Xs'],['Yr','Yp', 'Ys'])):
        _df = df.loc[:calicut]
        axes=sup_axes
        _df.plot.scatter(x=x, y=y, c=list(unique_cm_dict_from_list(_df.index.tolist()).values()),
                        ax=axes[i])


        if calicut is not None:
            _df = df.loc[calicut:]
            axes=sup_axes
            _df.plot.scatter(x=x, y=y, c=list(unique_cm_dict_from_list(_df.index.tolist()).values()),
                             ax=axes[1][i])

    if calicut is not None:
        axes = sup_axes
    else:
        axes = [sup_axes]

    for row, _axes in enumerate(axes):
        for col, ax in enumerate(_axes):
            if col is not 2:
                if xlim is not None:
                    ax.set_xlim(0,xlim)
                if ylim is not None:
                    ax.set_ylim(0,ylim)
            else:
                if xslim is not None:
                    ax.set_xlim(0, xslim)
                if yslim is not None:
                    ax.set_ylim(0, yslim)
            if invert_y:
                ax.invert_yaxis()

    pad=5
    for ax, col in zip(axes[0], ['Raw','Processed','Scaled']):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    if calicut is not None:
        for ax, row in zip(axes[:, 0], ['Calibration','Post-Calibration']):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation='vertical')

    fig.suptitle(xlabscsv)
    plt.show()




def main():
    parser=build_argparser()
    args = parser.parse_args()
    go(**vars(args))


