from matplotlib import pyplot as plt


def fixed_ax_aspect_ratio(ax: plt.axes, ratio: float) -> None:
    """
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    """
    xvals, yvals = ax.get_xlim(), ax.get_ylim()

    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')
