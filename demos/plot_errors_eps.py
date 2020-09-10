
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
import matplotlib.ticker as ticker
from shapely.geometry.polygon import LinearRing, Polygon

rc('text', usetex=True)
rc('font', family='serif')


plt.style.use('seaborn-colorblind')

def set_size(width, myratio, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * myratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width = 452.9679
fig = plt.figure(figsize=set_size(width, 0.3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

directory = "../build/demos/FiniteVolume-MR/d1q2_eps/"


eps             = np.loadtxt(directory +             "eps_s_1.000000_eps_0.100000.dat")

diff_ref_adap_1   = np.loadtxt(directory +   "diff_ref_adap_s_0.750000_eps_0.100000.dat")
compression_1     = np.loadtxt(directory +     "compression_s_0.750000_eps_0.100000.dat")

diff_ref_adap_2   = np.loadtxt(directory +   "diff_ref_adap_s_1.000000_eps_0.100000.dat")
compression_2     = np.loadtxt(directory +     "compression_s_1.000000_eps_0.100000.dat")

diff_ref_adap_3   = np.loadtxt(directory +   "diff_ref_adap_s_1.500000_eps_0.100000.dat")
compression_3     = np.loadtxt(directory +     "compression_s_1.500000_eps_0.100000.dat")

diff_ref_adap_4   = np.loadtxt(directory +   "diff_ref_adap_s_1.750000_eps_0.100000.dat")
compression_4     = np.loadtxt(directory +     "compression_s_1.750000_eps_0.100000.dat")


ax1.loglog(eps, diff_ref_adap_1, '.', label = '$s = 0.75$')
ax1.loglog(eps, diff_ref_adap_2, '.', label = '$s = 1.00$')
ax1.loglog(eps, diff_ref_adap_3, '.', label = '$s = 1.50$')
ax1.loglog(eps, diff_ref_adap_4, '.', label = '$s = 1.75$')
#ax1.loglog(eps, eps, 'black', label = "$\\epsilon$")
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))


max_eps = max(eps)
min_eps = min(eps)
alpha = 0.1
min_err = min(diff_ref_adap_1)
poly = Polygon([(1.0e-1, 1.0e-11), (1.0e-1, 1.0e-8), (1.0e-4, 1.0e-11)])
x,y = poly.exterior.xy

D = np.sqrt(1.0e-1 + 1.0e-4)
betabar = 10**(-5.0/(3.0))
betabar2 = 10**(-19.0/2.0)

print(betabar)

ax1.plot(x, y, color='black', alpha=1.0,  linestyle='dashed', 
    linewidth=0.75, solid_capstyle='round', zorder=2)
ax1.loglog([x[1], x[2]], [y[1], y[2]], color = 'black')

ax1.text(betabar, betabar2, "$\\epsilon$")

ax1.legend(fontsize = 5)
ax1.set_xlabel("$\\epsilon$")
ax1.set_title("$e^{0, N}$")


ax2.semilogx(eps, compression_1, '.')
ax2.semilogx(eps, compression_2, '.')
ax2.semilogx(eps, compression_3, '.')
ax2.semilogx(eps, compression_4, '.')
ax2.set_title("Compression")
ax2.set_xlabel("$\\epsilon$")

ax3.loglog(compression_1, diff_ref_adap_1, '.')
ax3.loglog(compression_2, diff_ref_adap_2, '.')
ax3.loglog(compression_3, diff_ref_adap_3, '.')
ax3.loglog(compression_4, diff_ref_adap_4, '.')
ax3.set_title("$e^{0, N}$")
ax3.set_xlabel("Comp")

fig.tight_layout()

plt.show()
