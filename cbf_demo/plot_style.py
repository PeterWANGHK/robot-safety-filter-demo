"""Reusable SciencePlots defaults for this project's exported figures.

SciencePlots requires importing ``scienceplots`` before selecting its styles.
We combine science/no-latex/bright as documented upstream, so a TeX installation
is unnecessary.  Context use keeps these choices local to an exported figure.
Source: https://github.com/garrettj403/SciencePlots
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401: registers the upstream Matplotlib styles


MODE_COLORS = {
    'unfiltered': '#EE6677',
    'filter-only': '#4477AA',
    'hierarchy': '#228833',
}
MODE_LABELS = {
    'unfiltered': 'Unfiltered',
    'filter-only': 'CBF only',
    'hierarchy': 'Planner + CBF',
}
MODE_LINESTYLES = {'unfiltered': '--', 'filter-only': '-.', 'hierarchy': '-'}
OBSTACLE_COLOR = '#666666'
BUFFER_COLOR = '#CC9900'


@contextmanager
def scientific_style(extra=None):
    """SciencePlots style with legible sizing and a consistent safe palette."""
    settings = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titlepad': 9,
        'axes.linewidth': .75,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.formatter.useoffset': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'lines.linewidth': 1.7,
        'lines.markersize': 4,
        'grid.linewidth': .45,
        'grid.alpha': .25,
        'savefig.dpi': 250,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    }
    if extra:
        settings.update(extra)
    with plt.style.context(['science', 'no-latex', 'bright', settings]):
        yield


def save_figure(figure, destination, *, dpi=250):
    """Write matching PNG and vector PDF files, returning their paths.

    ``destination`` may be a PNG/PDF filename or a suffix-free basename.  This
    deliberately exports both formats for every scientific figure in the demo.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    basename = destination.with_suffix('') if destination.suffix.lower() in {'.png', '.pdf'} else destination
    png, pdf = basename.with_suffix('.png'), basename.with_suffix('.pdf')
    figure.savefig(png, dpi=dpi, bbox_inches='tight', pad_inches=.09,
                   metadata={'Software': 'reBot Poisson CBF; SciencePlots'})
    figure.savefig(pdf, bbox_inches='tight', pad_inches=.09,
                   metadata={'Creator': 'reBot Poisson CBF; SciencePlots'})
    return png, pdf
