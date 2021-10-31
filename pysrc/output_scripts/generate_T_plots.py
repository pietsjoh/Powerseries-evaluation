def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
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


    textboxFormat = dict(boxstyle='square', facecolor='white')
    textboxStr = "\n".join((
        r"T$=20\,$K",
        r"Pump $\lambda = 785\,$nm"
    ))



    headDir = Path(__file__).resolve().parents[2]
    filePathTSeries = (headDir / "output" / "T-series.csv").resolve()
    dfTSeries = pd.read_csv(filePathTSeries, sep="\t", comment="#")
    temperature = dfTSeries["Temperature"]
    threshold = dfTSeries["threshold"]
    qFactor = dfTSeries["Q-factor"]
    qFactorUnc = dfTSeries["unc Q-factor"]
    betaBootstrap = dfTSeries["betaBt"]*100
    betaBootstrapUnc = dfTSeries["unc betaBt"]*100
    betaFixedXi = dfTSeries["betaFix"]*100
    betaFixedXiUnc = dfTSeries["unc betaFix"]*100
    modeEnergy = dfTSeries["mode energy"]
    modeEnergyUnc = dfTSeries["unc mode energy"]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    threshold1, = ax1.plot(temperature[:-4], threshold[:-4], lw=0, marker="^", markersize=5, color="black", label="Laserschwelle 1. Peak")
    threshold2, = ax1.plot(temperature[-4:], threshold[-4:], lw=0, marker="s", markersize=5, color="black", label="Laserschwelle 2. Peak")
    ax1.set_ylabel("Laserschwelle [mW]")
    ax1.set_xlabel("Temperatur [K]")
    ax1.locator_params(axis="x", nbins=12)
    ax1.set_xlim(10, 190)
    ax1.set_ylim(0, None)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    modeenenergy1 = ax2.errorbar(temperature[:-4], modeEnergy[:-4], yerr=modeEnergyUnc[:-4], fmt=".", marker="^",
                markersize=5, color="red", capsize=2.5, elinewidth=0.8, label="Modenergie 1. Peak")
    modeenergy2 = ax2.errorbar(temperature[-4:], modeEnergy[-4:], yerr=modeEnergyUnc[-4:], fmt=".", marker="s",
                markersize=5, color="red", capsize=2.5, elinewidth=0.8, label="Modenenergie 2. Peak")
    ax2.set_ylabel("Modenenergie [eV]")
    ax2.axvline(130, color="black", lw=1.1)
    ax2.axvline(140, color="black", lw=1.1)
    ax2.locator_params(axis="x", nbins=12)
    ax2.set_ylim(1.28, 1.32)
    ax2.spines["right"].set_color("red")
    ax2.yaxis.label.set_color("red")
    ax2.tick_params(axis="y", color="red", which="both")
    plt.setp(ax2.get_yticklabels(), color="red")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax1.text(132, 0.8, "mode switching", rotation="vertical", fontsize=12)
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"Durchmesser$=4\,\mu$m")
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Pump $\lambda = 785\,$nm")
    # plt.legend()
    legend = ax1.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.01, 0.5), loc="center left", fontsize=12,
            handles=[extra1, extra2, threshold1, threshold2, modeenenergy1, modeenergy2],
            labels=[r"Durchmesser$=4\,\mu$m", "Pump $\lambda = 785\,$nm", "Laserschwelle 1. Peak",
            "Laserschwelle 2. Peak", "Modenergie 1. Peak", "Modenenergie 2. Peak"])
    legend.get_frame().set_linewidth(0.5)
    # plt.savefig("out.png", dpi=1000)
    plt.tight_layout()
    plt.show()



    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    qFactor1 = ax1.errorbar(temperature[:-4], qFactor[:-4], yerr=qFactorUnc[:-4],
            fmt=".", marker="^", markersize=5, capsize=2.5, elinewidth=0.8, color="black", label="Q-Faktor 1. Peak")
    qFactor2 = ax1.errorbar(temperature[-4:], qFactor[-4:], yerr=qFactorUnc[-4:],
            fmt=".", marker="s", markersize=5, capsize=2.5, elinewidth=0.8, color="black", label="Q-Faktor 2. Peak")
    ax1.set_ylabel("Q-Faktor")
    ax1.set_xlabel("Temperatur [K]")
    ax1.locator_params(axis="x", nbins=12)
    ax1.set_xlim(10, 190)
    ax1.set_ylim(3000, 12500)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    beta1 = ax2.errorbar(temperature[:-4], betaFixedXi[:-4], yerr=betaFixedXiUnc[:-4], fmt=".", marker="^",
                markersize=5, color="red", capsize=2.5, elinewidth=0.8, label=r"$\beta$-Faktor 1. Peak")
    beta2 = ax2.errorbar(temperature[-4:], betaFixedXi[-4:], yerr=betaFixedXiUnc[-4:], fmt=".", marker="s",
                markersize=5, color="red", capsize=2.5, elinewidth=0.8, label=r"$\beta$-Faktor 2. Peak")
    ax2.set_ylabel(r"$\beta$-Faktor [%]")
    ax2.axvline(130, color="black", lw=1.1)
    ax2.axvline(140, color="black", lw=1.1)
    ax2.locator_params(axis="x", nbins=12)
    ax2.set_ylim(0, 19)
    ax2.spines["right"].set_color("red")
    ax2.yaxis.label.set_color("red")
    ax2.tick_params(axis="y", color="red", which="both")
    plt.setp(ax2.get_yticklabels(), color="red")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax1.text(132, 6300, "mode switching", rotation="vertical", fontsize=12)
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"Durchmesser$=4\,\mu$m")
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Pump $\lambda = 785\,$nm")
    # plt.legend()
    legend = ax1.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.445, 0.98), loc="upper center",
            fontsize=10, labelspacing=0.3, handletextpad=0.3, borderpad=0.3,
            handles=[extra1, extra2, qFactor1, qFactor2, beta1, beta2],
            labels=[r"Durchmesser$=4\,\mu$m", "Pump $\lambda = 785\,$nm", "Q-Faktor 1. Peak",
            "Q-Faktor 2. Peak", r"$\beta$-Faktor 1. Peak", r"$\beta$-Faktor 2. Peak"])
    legend.get_frame().set_linewidth(0.5)
    # plt.savefig("out.png", dpi=1000)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()