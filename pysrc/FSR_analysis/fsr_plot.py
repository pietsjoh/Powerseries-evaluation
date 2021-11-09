from scipy.sparse import dia


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, cycler
    from matplotlib.ticker import AutoMinorLocator
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
    rcParams.update({"figure.figsize" : (6.4,4.8),
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

    HeadDir = Path(__file__).resolve().parents[2]
    srcDirPath = (HeadDir / "pysrc").resolve()
    sys.path.append(str(srcDirPath))

    import utils.misc as misc

    fsrCsvPath = (HeadDir / "output" / "fsr.csv").resolve()
    fsrMeanPath = (HeadDir / "output" / "fsr_mean.csv").resolve()

    data = pd.read_csv(fsrCsvPath, sep="\t")
    diameterList = data.loc[0].to_numpy()
    fsrData = data.drop([0], axis=0)

    dictFSR = {str(i) : [] for i in diameterList}
    fsrList = []
    for i, col in enumerate(fsrData.columns):
        fsr = fsrData[col].to_numpy()
        dictFSR[str(diameterList[i])] = [*dictFSR[str(diameterList[i])], *fsr]
    # print(dictFSR["3.0"])
    diameterList = []
    meanList = []
    uncList = []
    for key in dictFSR.keys():
        values = np.array(dictFSR[key])
        values = values[~np.isnan(values)]
        mean = np.mean(values)
        uncMean = misc.unc_mean(values)
        diameterList.append(float(key))
        meanList.append(mean)
        uncList.append(uncMean)

    dfDict = {"diameter" : diameterList, "mean" : meanList, "unc" : uncList}
    df = pd.DataFrame(dfDict).set_index("diameter")
    # print(df)
    # df.to_csv(fsrMeanPath, sep="\t")

    def fsr(d, A):
        return 1 / (d*A)

    def fsr_inv(d, A):
        return A*d

    import scipy.optimize as optimize

    idx_25 = diameterList.index(2)
    diameterList_wo_25 = [d for i, d in enumerate(diameterList) if i != idx_25]
    meanList_wo_25 = [d for i, d in enumerate(meanList) if i != idx_25]
    p, cov = optimize.curve_fit(fsr, diameterList_wo_25, meanList_wo_25)
    p_inv, cov_inv = optimize.curve_fit(fsr_inv, 1 / np.array(diameterList_wo_25), meanList_wo_25)
    print(p*1000, np.sqrt(cov)*1000)
    print(p_inv*1000, np.sqrt(cov_inv)*1000)
    plotArr = np.linspace(2, 20, 1000)

    textboxFormat = dict(boxstyle='square', facecolor='white', linewidth=0.5)
    textboxStr = "\n".join((
        r"T$=20\,$K",
        r"Pump $\lambda = 785\,$nm"
    ))
    fig, ax = plt.subplots()
    # ax.errorbar(diameterList, meanList, yerr=uncList, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, label="Messpunkte", color="black")
    # ax.plot(plotArr, fsr(plotArr, *p), label=r"Fit: FSR$=1/(A\cdot d)$", color="blue")
    ax.errorbar(1 / np.array(diameterList), meanList, yerr=uncList, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5, label="Messpunkte", color="black")
    ax.plot(1 / plotArr, fsr(plotArr, *p), label=r"Fit: FSR$=1/(A\cdot d)$", color="blue")
    ax.plot(1 / plotArr, fsr_inv(1 / plotArr, *p_inv), label=r"Fit: FSR$=A\cdot d)$", color="yellow")
    ax.set_xlabel(r"Durchmesser $d$ [µm]")
    ax.set_ylabel(r"FSR $\Delta\lambda$ ([nm]")
    ax.locator_params(axis="x", nbins=12)
    # ax.set_xlim(0, 21)
    # ax.set_ylim(0, 27)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 27)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.text(13, 25, textboxStr, fontsize=13, va="top", bbox=textboxFormat)
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"T$=20\,$K")
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Pump $\lambda = 785\,$nm")
    extra3 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"A$=(14.91\pm 0.15)\,$µm$^{-2}$")
    handles, labels = ax.get_legend_handles_labels()
    handles_list = [extra1, extra2] + handles[::-1] + [extra3]
    labels_list = [r"T$=20\,$K", "Pump $\lambda = 785\,$nm"] + labels[::-1] + [r"A$=(14.91\pm 0.15)\,$µm$^{-2}$"]
    legend = ax.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.4, 0.75), loc="center left", fontsize=12,
            handles=handles_list, labels=labels_list)
    legend.get_frame().set_linewidth(0.5)
    # plt.savefig("out.png", dpi=1000)
    plt.show()
