
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
plt.rcParams['image.cmap'] = 'gist_ncar'  # change default colormap

import matplotlib.patches as patches
import numpy as np


def unique_cm_dict_from_list(items, basecmap = 'gist_rainbow'):
    # From Sacrum
    cm = plt.get_cmap(basecmap)
    cnorm = mpl.colors.Normalize(vmin=0, vmax=len(items))
    scalarmap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cm)
    return dict(zip(items, [scalarmap.to_rgba(i) for i in range(len(items))])), scalarmap

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlabscsv", help="XLabs Output CSV for Visualisation", nargs='+')
    parser.add_argument("-c", "--calicut", action='store', default=None, type=str, help="Estimated index of the end of Calibration, can be list of x,y,z...")
    parser.add_argument("-x", "--xlim", action='store', default=1920, help="Restrict view to given x limit")
    parser.add_argument("-y", "--ylim", action='store', default=1080, help="Restrict view to given y limit")
    parser.add_argument("--xslim", action='store', default=1, help="Restrict view to given x limit for the scaled view")
    parser.add_argument("--yslim", action='store', default=1, help="Restrict view to given y limit for the scaled view")
    parser.add_argument("-Y", "--invert_y", action='store_true', default=False, help="Invert y-axis")
    parser.add_argument("-C", "--confidence_factor", action='store', default=1.0, type=float, help='Limit output to given percentage confidence (i.e. 0.8 drops 20%% worst)')
    parser.add_argument("-f", "--figure", action='store_true', default=False, help='Don\'t show the figure; dump it to "xlabscsv.png"')

    return parser

def go(xlabscsv, calicut=None, invert_y=False, xlim=None, ylim=None, xslim=None, yslim=None,
       confidence_factor=None, figure=False, **kwargs):
    if isinstance(xlabscsv, list):
        df = pd.concat([
                           pd.read_csv(xlabs,
                                       names=["Xr", "Yr", "Xp", "Xs", "Yp", "Ys", "C", 3, ],
                                       index_col=False)
                           for xlabs in xlabscsv
                           ])
    else:
        df = pd.read_csv(xlabscsv,
                         names=["Xr", "Yr", "Xp", "Xs", "Yp", "Ys", "C", 3, ],
                         index_col=False)
    df.reset_index(inplace=True)

    if calicut is None:
        cuts = [0]
    elif calicut.isdigit():
        cuts = [int(calicut)]
    elif ',' in calicut:
        cuts = list(map(int, calicut.split(',')))
    else:
        raise NotImplementedError("No idea how to deal with calicuts: {}".format(calicut))

    # Goes from [x,y,z] to [(0,x-1),(x,y-1),(y,z-1),(z,N)]
    split_indexes = list(map(lambda a: (a.min(),a.max()),np.array_split(df['index'], cuts)))
    try:
        split_indexes.remove((np.nan, np.nan))
    except ValueError:
        pass
    except:
        raise

    colorlist, scalarmap = unique_cm_dict_from_list(df['index'].unique().tolist())

    if confidence_factor is not None:
        conf_val = df.C.quantile(confidence_factor)
        df = df[df.C < conf_val]
    else:
        conf_val = None



    fig, axes = plt.subplots(nrows = len(split_indexes), ncols=3, figsize= (16,10))

    if len(split_indexes)==1:
        axes=[axes]

    fig.suptitle("{file},-Y={yinv},-C={C}".format(
        file=xlabscsv, yinv=invert_y, C=(confidence_factor,conf_val)
        )
    )
    fig.subplots_adjust(right=0.925)
    cbar_ax = fig.add_axes([0.95,0.05,0.01,0.9])
    scalarmap._A = df['index'].values

    for i, (lower,upper) in enumerate(split_indexes):
        for j,(x,y) in enumerate(zip(['Xr','Xp', 'Xs'],['Yr','Yp', 'Ys'])):
            _df=df.loc[lower:upper]
            _df.plot.scatter(x=x, y=y, c=[scalarmap.to_rgba(c) for c in _df['index'].values], ax=axes[i][j])


    for row, _axes in enumerate(axes):
        for col, ax in enumerate(_axes):
            if col == 3: # Colormap
                pass
            if col == 2: # Scaled
                if xslim is not None:
                    ax.set_xlim(0, xslim)
                if yslim is not None:
                    ax.set_ylim(0, yslim)

            else:
                if xlim is not None:
                    ax.set_xlim(0,xlim)
                if ylim is not None:
                    ax.set_ylim(0,ylim)
            if invert_y and col < 3:
                ax.invert_yaxis()

    pad=5
    for ax, col in zip(axes[0], ['Raw','Processed','Scaled']):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    if len(split_indexes)>1:
        for ax, row in zip(axes[:, 0], ["{} to {}".format(lower, upper) for lower,upper in split_indexes]):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=320)


    scalarmap._A=[]
    plt.colorbar(scalarmap, cax=cbar_ax)
    cbar_ax.invert_yaxis()
    if figure:
        fig.savefig(os.path.join(xlabscsv,'.png'))
    else:
        plt.show()




def main():
    parser=build_argparser()
    args = parser.parse_args()
    go(**vars(args))


