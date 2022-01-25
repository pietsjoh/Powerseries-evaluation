def main():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, cycler
    from matplotlib.ticker import AutoMinorLocator
    import scipy.ndimage as ndi

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
    outputDir = (headDir / "output_plots" / "diameter").resolve()
    filePathThreshold = (outputDir / "threshold_mean.csv").resolve()
    filePathQFactor = (outputDir / "Q-factor_mean.csv").resolve()
    # filePathbetaBootstrap = (outputDir / "beta_bt_all.csv").resolve()
    filePathbetaBootstrap = (outputDir / "beta_bt_mean.csv").resolve()
    # filePathbetaBootstrap = (outputDir / "beta_bt_weighted_mean.csv").resolve()
    # filePathbetaFixedXi = (outputDir / "beta_fix_all.csv").resolve()
    filePathbetaFixedXi = (outputDir / "beta_fix_mean.csv").resolve()
    # filePathbetaFixedXi = (outputDir / "beta_fix_weighted_mean.csv").resolve()

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

    dfBetaBootstrap = pd.read_csv(filePathbetaBootstrap, sep="\t", comment="#")
    diameterBetaBootstrap = dfBetaBootstrap["diameter"].to_numpy()
    # betaBootstrap = dfBetaBootstrap["weighted mean"]*100
    betaBootstrap = dfBetaBootstrap["mean"]*100
    # betaBootstrap = dfBetaBootstrap["beta"]*100
    uncBetaBootstrap = dfBetaBootstrap["unc"]*100

    dfBetaFixedXi = pd.read_csv(filePathbetaFixedXi, sep="\t", comment="#")
    diameterBetaFixedXi = dfBetaFixedXi["diameter"]
    # betaFixedXi = dfBetaFixedXi["weighted mean"]*100
    betaFixedXi = dfBetaFixedXi["mean"]*100
    # betaFixedXi = dfBetaFixedXi["beta"]*100
    uncBetaBFixedXi = dfBetaFixedXi["unc"]*100



    # smoothThreshold = ndi.gaussian_filter1d(threshold.to_numpy(), sigma=3)
    # plt.errorbar(diameterThreshold, threshold, yerr=uncThreshold, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    # plt.plot(diameterThreshold, smoothThreshold, color="black")
    # plt.ylabel("Laserschwelle [mW]")
    # plt.xlabel("Durchmesser [µm]")
    # plt.locator_params(axis="x", nbins=12)
    # plt.xlim(0, 21)
    # plt.ylim(0, 1.25)
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.text(1, 1.17, textboxStr, fontsize=13, va="top", bbox=textboxFormat)
    # # plt.savefig("out.png", dpi=1000)
    # plt.show()



    # smoothQFactor = ndi.gaussian_filter1d(Qfactor.to_numpy(), sigma=2)

    # plt.errorbar(diameterQfactor, Qfactor, yerr=uncQfactor, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    # plt.plot(diameterQfactor, smoothQFactor, color="black")
    # plt.ylabel("Q-Faktor")
    # plt.xlabel("Durchmesser [µm]")
    # plt.locator_params(axis="x", nbins=12)
    # plt.xlim(0, 21)
    # plt.ylim(2000, 20000)
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.text(1, 18000, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # # plt.savefig("out.png", dpi=1000)
    # plt.show()


    # smoothBetaBt = ndi.gaussian_filter1d(betaBootstrap.to_numpy(), sigma=10)
    smoothBetaBt = ndi.gaussian_filter1d(betaBootstrap.to_numpy(), sigma=3)
    plt.errorbar(diameterBetaBootstrap, betaBootstrap, yerr=uncBetaBootstrap, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    plt.plot(diameterBetaBootstrap, smoothBetaBt, color="black")
    # plt.yscale("log")
    plt.ylabel(r"$\beta$-Faktor [%]")
    plt.xlabel("Durchmesser [µm]")
    plt.locator_params(axis="x", nbins=12)
    plt.xlim(0, 21)
    # plt.ylim(0, 3)
    plt.ylim(-1.9, None)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.text(13.5, 10.5, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # plt.savefig("out.png", dpi=1000)
    plt.show()



    # smoothBetaFix = ndi.gaussian_filter1d(betaFixedXi, sigma=10)
    # smoothBetaFix = ndi.gaussian_filter1d(betaFixedXi, sigma=3)
    # plt.errorbar(diameterBetaFixedXi, betaFixedXi, yerr=uncBetaBFixedXi, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    # plt.plot(diameterBetaFixedXi, smoothBetaFix, color="black")
    # plt.ylabel(r"$\beta$-Faktor [%]")
    # plt.xlabel("Durchmesser [µm]")
    # plt.locator_params(axis="x", nbins=12)
    # plt.xlim(0, 21)
    # # plt.ylim(0, 1.5)
    # plt.ylim(0, None)
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # # plt.text(13.5, 1.33, textboxStr, fontsize=13, va="center", bbox=textboxFormat)
    # # plt.savefig("out.png", dpi=1000)
    # plt.show()



    def V_M(d):
        return 5.8*d**(1.3) / 0.9

    diameterPlotArr = np.linspace(0, 20, 1000)
    VMDiameter = V_M(diameterPlotArr)
    VMDiameterQFactor = V_M(diameterQfactor.to_numpy())
    QdivV = Qfactor / VMDiameterQFactor
    uncQdivV = uncQfactor / VMDiameterQFactor

    smoothQdivV = ndi.gaussian_filter1d(QdivV, sigma=2)

    betaEstimate = 100 / (1 + 1000/QdivV)
    smoothBetaEstimate = ndi.gaussian_filter1d(betaEstimate, sigma=2)
    # plt.errorbar(diameterQfactor, QdivV, yerr=uncQdivV, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    # plt.plot(diameterQfactor, betaEstimate, color="black", marker="s", markersize=5, linewidth=0)
    # plt.plot(diameterQfactor, smoothBetaEstimate, color="black")
    # plt.xlim(0, 21)
    # plt.show()

    # _, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.errorbar(diameterQfactor, QdivV, yerr=uncQdivV, capsize=2.5, elinewidth=0.8, fmt=".", marker="s", markersize=5)
    # ax1.plot(diameterQfactor, smoothQdivV, color="black")
    # ax2.plot(diameterPlotArr, VMDiameter, color="red")
    # ax1.locator_params(axis="x", nbins=12)
    # ax1.set_xlim(0, 21)
    # ax1.set_ylim(0, None)
    # ax1.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # ax2.locator_params(axis="x", nbins=12)
    # ax2.set_ylim(0, None)
    # ax2.spines["right"].set_color("red")
    # ax2.yaxis.label.set_color("red")
    # ax2.tick_params(axis="y", color="red", which="both")
    # plt.setp(ax2.get_yticklabels(), color="red")
    # ax2.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    # ax2.set_ylabel(r"Modenvolumen [$(\lambda /n)^3$]")
    # ax1.set_xlabel("Durchmesser [µm]")
    # ax1.set_ylabel(r"$Q/V_M$ [$(n / \lambda)^3$]")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()