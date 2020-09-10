
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
import matplotlib.ticker as ticker

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

directory = "../build/demos/FiniteVolume-MR/d1q2/"

time_frames = np.loadtxt(directory + "time_frame_s_1.000000_eps_0.000100.dat")

error_exact_ref_1 = np.loadtxt(directory + "error_exact_ref_0.750000_eps_0.000100.dat")
diff_ref_adap_1   = np.loadtxt(directory +   "diff_ref_adap_s_0.750000_eps_0.000100.dat")
compression_1     = np.loadtxt(directory +     "compression_s_0.750000_eps_0.000100.dat")

error_exact_ref_2 = np.loadtxt(directory + "error_exact_ref_1.000000_eps_0.000100.dat")
diff_ref_adap_2   = np.loadtxt(directory +   "diff_ref_adap_s_1.000000_eps_0.000100.dat")
compression_2     = np.loadtxt(directory +     "compression_s_1.000000_eps_0.000100.dat")

# dt = time_frames[1] - time_frames[0]
# derivative = (diff_ref_adap_2[1:len(time_frames)] - diff_ref_adap_2[0:len(time_frames) - 1]) / dt
# print(np.mean(derivative) / 1.0e-4)

# foo   = np.loadtxt(directory +   "diff_ref_adap_s_1.000000_eps_0.000000.dat")
# derivativefoo = (foo[1:len(time_frames)] - foo[0:len(time_frames) - 1]) / dt
# print(np.mean(derivativefoo) / 1.0e-8)


error_exact_ref_3 = np.loadtxt(directory + "error_exact_ref_1.500000_eps_0.000100.dat")
diff_ref_adap_3   = np.loadtxt(directory +   "diff_ref_adap_s_1.500000_eps_0.000100.dat")
compression_3     = np.loadtxt(directory +     "compression_s_1.500000_eps_0.000100.dat")


error_exact_ref_4 = np.loadtxt(directory + "error_exact_ref_1.750000_eps_0.000100.dat")
diff_ref_adap_4   = np.loadtxt(directory +   "diff_ref_adap_s_1.750000_eps_0.000100.dat")
compression_4     = np.loadtxt(directory +     "compression_s_1.750000_eps_0.000100.dat")

const = 1.6

ax1.plot(range(len(time_frames)),  diff_ref_adap_1, label = "$s = 0.75$")
ax1.plot(range(len(time_frames)),  diff_ref_adap_2, label = "$s = 1.00$")
ax1.plot(range(len(time_frames)),  diff_ref_adap_3, label = "$s = 1.50$")
ax1.plot(range(len(time_frames)),  diff_ref_adap_4, label = "$s = 1.75$")
#ax1.plot(range(len(time_frames)),  const * 0.0001*(np.arange(len(time_frames)) +  + np.ones(len(time_frames))), label = "bound")
ax1.legend(fontsize = 5)
ax1.set_xlabel("$n$")
ax1.set_title("$e^{0, n}$")
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
# ax1.plot(time_frames,  diff_ref_adap_5)

ax2.semilogy(range(len(time_frames)),  error_exact_ref_1 / diff_ref_adap_1)
ax2.semilogy(range(len(time_frames)),  error_exact_ref_2 / diff_ref_adap_2)
ax2.semilogy(range(len(time_frames)),  error_exact_ref_3 / diff_ref_adap_3)
ax2.semilogy(range(len(time_frames)),  error_exact_ref_4 / diff_ref_adap_4)
# ax2.plot(time_frames,  error_exact_ref_5 / diff_ref_adap_5)
ax2.set_xlabel("$n$")
ax2.set_title("${E^{0, n}} / {e^{0, n}}$")


ax3.plot(range(len(time_frames)),  compression_1, '.')
ax3.plot(range(len(time_frames)),  compression_2, '.')
ax3.plot(range(len(time_frames)),  compression_3, '.')
ax3.plot(range(len(time_frames)),  compression_4, '.')
ax3.set_xlabel("$n$")
#ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
ax3.ticklabel_format(axis = 'y', style = 'sci')
ax3.set_title("Compression")



fig.tight_layout()

plt.show()
