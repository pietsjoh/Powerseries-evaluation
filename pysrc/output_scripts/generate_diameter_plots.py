def main():
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, cycler
    from matplotlib.ticker import AutoMinorLocator

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


    headDir = Path(__file__).resolve().parents[2]
    filePathThreshold = (headDir / "output" / "threshold_mean.csv").resolve()
    filePathQFactor = (headDir / "output" / "Q-factor_mean.csv").resolve()
    filePathbetaBootstrap = (headDir / "output" / "beta_bt_mean.csv").resolve()
    filePathbetaFixedXi = (headDir / "output" / "beta_fix_mean.csv").resolve()

    textboxFormat = dict(boxstyle='square', facecolor='white', linewidth=0.5)
    textboxStr = "\n".join((
        r"T$=20\,$K",
        r"Pump $\lambda = 785\,$nm"
    ))

    dfThreshold = pd.read_csv(filePathThreshold, sep="\t", comment="#")
    diameterThreshold = dfThreshold["diameter"]
    threshold = dfThreshold["mean"]
    uncThreshold = dfThreshold["unc"]

    dfQfactor = pd.read_csv(filePathQFactor, sep="\t", comment="#")
    diameterQfactor = dfQfactor["diameter"]
    Qfactor = dfQfactor["weighted mean"]
    uncQfactor = dfQfactor["unc"]

    dfBetaBootstrap = pd.read_csv(filePathbetaBootstrap, sep="\t")
    diameterBetaBootstrap = dfBetaBootstrap["diameter"].to_numpy()
    betaBootstrap = dfBetaBootstrap["weighted mean"]*100
    uncBetaBootstrap = dfBetaBootstrap["unc"]*100

    dfBetaFixedXi = pd.read_csv(filePathbetaFixedXi, sep="\t", comment="#")
    diameterBetaFixedXi = dfBetaFixedXi["diameter"]
    betaFixedXi = dfBetaFixedXi["weighted mean"]*100
    uncBetaBFixedXi = dfBetaFixedXi["unc"]*100

    plt.errorbar(diameterThreshold, threshold, yerr=uncThreshold, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    plt.ylabel("Laserschwelle [mW]")
    plt.xlabel("Durchmesser [µm]")
    plt.locator_params(axis="x", nbins=12)
    plt.xlim(0, 21)
    plt.ylim(0, 1.25)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.text(1, 1.17, textboxStr, fontsize=13, va="top", bbox=textboxFormat)
    # plt.savefig("out.png", dpi=1000)
    plt.show()

    plt.errorbar(diameterQfactor, Qfactor, yerr=uncQfactor, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    plt.ylabel("Q-Faktor")
    plt.xlabel("Durchmesser [µm]")
    plt.locator_params(axis="x", nbins=12)
    plt.xlim(0, 21)
    plt.ylim(2000, 20000)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.text(1, 18000, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # plt.savefig("out.png", dpi=1000)
    plt.show()

    plt.errorbar(diameterBetaBootstrap, betaBootstrap, yerr=uncBetaBootstrap, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    plt.ylabel(r"$\beta$-Faktor [%]")
    plt.xlabel("Durchmesser [µm]")
    plt.locator_params(axis="x", nbins=12)
    plt.xlim(0, 21)
    plt.ylim(0, 3)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.text(13.5, 2.65, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # plt.savefig("out.png", dpi=1000)
    plt.show()

    plt.errorbar(diameterBetaFixedXi, betaFixedXi, yerr=uncBetaBFixedXi, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    plt.ylabel(r"$\beta$-Faktor [%]")
    plt.xlabel("Durchmesser [µm]")
    plt.locator_params(axis="x", nbins=12)
    plt.xlim(0, 21)
    plt.ylim(0, 1.5)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.text(13.5, 1.33, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # plt.savefig("out.png", dpi=1000)
    plt.show()

if __name__ == "__main__":
    main()