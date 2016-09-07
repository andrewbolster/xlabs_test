
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
    return dict(zip(items, [scalarmap.to_rgba(i) for i in range(len(items))])), scalarmap

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlabscsv", help="XLabs Output CSV for Visualisation")
    parser.add_argument("-c", "--calicut", action='store', default=None, type=str, help="Estimated index of the end of Calibration, can be list of x,y,z...")
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

    if calicut is None:
        cuts = [0]
    elif calicut.isdigit():
        cuts = [int(calicut)]
    elif ',' in calicut:
        cuts = map(int, calicut.split(','))
    else:
        raise NotImplementedError("No idea how to deal with calicuts: {}".format(calicut))

    # Goes from [x,y,z] to [(0,x-1),(x,y-1),(y,z-1),(z,N)]
    split_indexes = list(map(lambda a: (a.min(),a.max()),np.array_split(df.index, cuts)))
    try:
        split_indexes.remove((np.nan, np.nan))
    except ValueError:
        pass
    except:
        raise

    if confidence_factor is not None:
        df = df[df.C < df.C.quantile(confidence_factor)]

    colorlist, scalarmap = unique_cm_dict_from_list(df.index.tolist())
    df['color'] = colorlist

    fig, axes = plt.subplots(nrows = len(split_indexes), ncols=4, figsize= (16,10), gridspec_kw=dict(width_ratios=[3,3,3,1]))

    if len(split_indexes)==1:
        axes=[axes]

    for i, (lower,upper) in enumerate(split_indexes):
        for j,(x,y) in enumerate(zip(['Xr','Xp', 'Xs'],['Yr','Yp', 'Ys'])):
            df.loc[lower:upper].plot.scatter(x=x, y=y, c='index', ax=axes[i][j], colorbar=True)


    for row, _axes in enumerate(axes):
        for col, ax in enumerate(_axes):
            if col == 3: # Colormap
                pass
            if col == 2: # Scaled
                if xslim is not None:
                    ax.set_xlim(0, xslim)
                if yslim is not None:
                    ax.set_ylim(0, yslim)

                cmap_base_ax = ax # Keep score axis figure for colourmap generation
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
                        size='large', ha='right', va='center', rotation='vertical')

    fig.suptitle(xlabscsv)
    plt.show()




def main():
    parser=build_argparser()
    args = parser.parse_args()
    go(**vars(args))


