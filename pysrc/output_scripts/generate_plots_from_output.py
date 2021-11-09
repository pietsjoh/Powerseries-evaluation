import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, ticker
from matplotlib.patches import Rectangle

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 15
rcParams['axes.linewidth'] = 1.1
rcParams['axes.labelpad'] = 10.0
plot_color_cycle = cycler('color', ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b', 
                                    'e377c2', '7f7f7f', 'bcbd22', '17becf'])
rcParams['axes.prop_cycle'] = plot_color_cycle
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
rcParams.update({"figure.figsize" : (6.4, 4.8), #6.6, 5
                     "figure.subplot.left" : 0.177, "figure.subplot.right" : 0.946,
                     "figure.subplot.bottom" : 0.156, "figure.subplot.top" : 0.965,
                     "axes.autolimit_mode" : "round_numbers",
                     "xtick.major.size"     : 7,
                     "xtick.minor.size"     : 3.5,
                     "xtick.major.width"    : 1.1,
                     "xtick.minor.width"    : 1.1,
                     "xtick.major.pad"      : 5,
                     "xtick.minor.visible" : True,
                     "ytick.major.size"     : 7,
                     "ytick.minor.size"     : 3.5,
                     "ytick.major.width"    : 1.1,
                     "ytick.minor.width"    : 1.1,
                     "ytick.major.pad"      : 5,
                     "ytick.minor.visible" : True,
                     "lines.markersize" : 5,
                    #  "lines.markerfacecolor" : "none",
                     "lines.markeredgewidth"  : 0.8,
                     "xtick.direction" : "in",
                     "ytick.direction" : "in",
                     "ytick.right" : True,
                     "xtick.top" : True
})

textboxFormat = dict(boxstyle='square', facecolor='white', linewidth=0.5)
textboxStr = "\n".join((
    r"T$=20\,$K",
    r"Pump $\lambda = 785\,$nm",
    r"$d = 4\,$µm"
))

headDirPath: Path = Path(__file__).parents[2]

outDirPath: Path = (headDirPath / "output").resolve()
filePathPowerseries: Path = (outDirPath / "powerseries.csv").resolve()
filePathBetaFit: Path = (outDirPath / "beta_fit.csv").resolve()
filePathSettings: Path = (outDirPath / "settings.csv").resolve()

dfPowerseries: pd.DataFrame = pd.read_csv(filePathPowerseries, sep="\t")
dfBetaFit: pd.DataFrame = pd.read_csv(filePathBetaFit, sep="\t")
# dfSettings: pd.DataFrame = pd.read_csv(filePathSettings, sep="\t")

Q = dfPowerseries["Q-factor"]
QUnc = dfPowerseries["uncQ-factor"]
lw = dfPowerseries["lw"]
lwUnc = dfPowerseries["uncLw"]
out = dfPowerseries["out"].to_numpy()
outUnc = dfPowerseries["uncOut"]
inp = dfPowerseries["in"].to_numpy()
E = dfPowerseries["modeEnergy"]
EUnc = dfPowerseries["uncModeEnergy"]

p = dfBetaFit["fitParamsBeta"]

def in_out_model(x, beta, p, A, xiHat):
    return (A / beta) * ( p*x / (p*x + 1) * (1 + xiHat*beta) * (1 + beta*p*x) - (beta**2)*xiHat*p*x)

fig, ax = plt.subplots()
ax.errorbar(inp, Q, yerr=QUnc, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
ax.set_xscale("log")
ax.set_ylabel("Q-Faktor")
ax.set_xlabel("Eingangsleistung [mW]")
ax.set_xlim(0.15, None)
ax.set_ylim(0, None)
ax.text(0.2, 55000, textboxStr, fontsize=13, va="top", bbox=textboxFormat)
plt.show()

fig, ax = plt.subplots()
ax.errorbar(inp, E, yerr=EUnc, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
ax.set_xscale("log")
ax.set_ylabel("Modenenergie [eV]")
ax.set_xlabel("Eingangsleistung [mW]")
ax.set_xlim(0.15, None)
ax.set_ylim(None, None)
ax.text(2, 1.32343, textboxStr, fontsize=13, va="top", bbox=textboxFormat)
plt.tight_layout()
plt.show()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# EPlot = ax2.errorbar(inp, E, yerr=EUnc, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, color="red", label="Modenenergie")
# ax2.set_xscale("log")
# ax2.set_ylabel("Modenenergie [eV]")
# ax2.set_xlabel("Eingangsleistung [mW]")
# ax2.set_xlim(0.15, None)
# ax2.set_ylim(None, None)
# QPlot = ax1.errorbar(inp, Q, yerr=QUnc, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, color="black", label="Q-Faktor")
# ax1.set_xscale("log")
# ax1.set_ylabel("Q-Faktor")
# ax1.set_xlabel("Eingangsleistung [mW]")
# ax1.set_xlim(0.15, None)
# ax1.set_ylim(0, None)
# ax2.spines["right"].set_color("red")
# ax2.yaxis.label.set_color("red")
# ax2.tick_params(axis="y", color="red", which="both")
# plt.setp(ax2.get_yticklabels(), color="red")
# extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"T$=20\,$K")
# extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Pump $\lambda = 785\,$nm")
# legend = ax1.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.99, 0.01), loc="lower right", fontsize=12,
#         handles=[extra1, extra2, QPlot, EPlot],
#         labels=[r"T$=20\,$K, $d=4\,\mu$m", "Pump $\lambda = 785\,$nm", "Modenenergie",
#         "Q-Faktor"])
# legend.get_frame().set_linewidth(0.5)
# plt.tight_layout()
# plt.show()

outPlotArr = np.logspace(np.log10(min(out)), np.log10(max(out)), 100)
inPlotArr = in_out_model(outPlotArr, *p)

minInpIdx = np.argmin(inp)
maxInpIdx = np.argmax(inp)
minInp = inp[minInpIdx]
maxInp = inp[maxInpIdx]
minOut = out[minInpIdx]
maxOut = out[maxInpIdx]

aLow = minOut / minInp
aHigh = maxOut / maxInp

lowInpPlot = np.logspace(np.log10(min(inp)), np.log10(1), 100)
highInpPlot = np.logspace(np.log10(1), np.log10(max(inp)), 100)
lowOutPlot = aLow * lowInpPlot
highOutPlot = aHigh * highInpPlot
# print(min(inp), minInp)
# print(max(inp), maxInp)
# print(min(out), minOut)
# print(max(out), maxOut)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.errorbar(inp, out, yerr=outUnc, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, color="black", label="Intensität")
ax1.plot(inPlotArr, outPlotArr, color="blue", label="Fit Intensität")
ax1.plot(lowInpPlot, lowOutPlot, color="green", lw="0.9")
ax1.plot(highInpPlot, highOutPlot, color="green", lw="0.9", label="lineare Funktion")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("PL Intensität [a. u.]")
ax1.set_xlabel("Eingangsleistung [mW]")
ax1.set_xlim(0.15, None)
ax1.set_ylim(None, 200)
lwPlot = ax2.errorbar(inp, lw*1000, yerr=lwUnc*1000, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, color="red", label="FWHM")
ax2.set_xscale("log")
ax2.set_ylabel("FWHM [meV]")
ax2.set_xlabel("Eingangsleistung [mW]")
ax2.set_xlim(0.15, None)
ax2.set_ylim(0.02, 0.24)
ax2.spines["right"].set_color("red")
ax2.yaxis.label.set_color("red")
ax2.tick_params(axis="y", color="red", which="both")
plt.setp(ax2.get_yticklabels(), color="red")
extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"T$=20\,$K")
extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Pump $\lambda = 785\,$nm")
# extra3 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"d$=4\,\mu$m")
handles, labels = ax1.get_legend_handles_labels()
handles_list = [extra1, extra2, lwPlot] + handles[::-1]
labels_list = [r"T$=20\,$K, $d=4\,\mu$m", "Pump $\lambda = 785\,$nm", "FWHM"] + labels[::-1]
legend = ax1.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.99, 0.15), loc="lower right", fontsize=12, handles=handles_list, labels=labels_list)
legend.get_frame().set_linewidth(0.5)
plt.tight_layout()
plt.show()

## todo: plot single spectrum with lorentz fit, evtl. linewidth fit