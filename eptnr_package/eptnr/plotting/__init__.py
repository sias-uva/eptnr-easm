from matplotlib import rcParams

params = {
    'axes.labelsize': 20,
    'axes.titlepad': 15,
    'axes.titlesize': 24,
    'font.size': 20,
    'legend.fontsize': 14,
    'axes.labelsize': 18,
    'text.usetex': False,
    'figure.figsize': [5, 5],
    'axes.labelpad': 16,
    'lines.linewidth': 2,
    "figure.edgecolor": "black",
    'axes.linewidth': 2,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'xtick.minor.pad': 6,
    'ytick.minor.pad': 6
}
rcParams['agg.path.chunksize'] = 10000
rcParams.update(params)
