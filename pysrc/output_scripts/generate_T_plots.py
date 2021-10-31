def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    headDir = Path(__file__).resolve().parents[2]
    filePathTSeries = (headDir / "output" / "T-series.csv").resolve()
    # filePathThreshold = (headDir / "output" / "Temp_threshold.csv").resolve()
    # filePathbetaBootstrap = (headDir / "output" / "Temp_beta_bootstrap.csv").resolve()
    # filePathbetaFixedXi = (headDir / "output" / "Temp_beta_fixed_xi.csv").resolve()
    # filePathbetaFixedXiNoQEstimate = (headDir / "output" / "Temp_beta_fixed_xi_no_q_estimate.csv").resolve()
    # filePathQFactor = (headDir / "output" / "Temp_Q-factor.csv").resolve()
    # filePathModeEnergy = (headDir / "output" / "Temp_mode_energy.csv")

    dfTSeries = pd.read_csv(filePathTSeries, sep="\t", comment="#")

    plt.errorbar(dfTSeries["Temperature"], dfTSeries["betaBt"], yerr=dfTSeries["unc betaBt"], fmt="bs", capsize=3)
    # plt.xlabel("Temperature [K]")
    # plt.ylabel("beta factor")
    # plt.show()
    plt.errorbar(dfTSeries["Temperature"], dfTSeries["betaFix"], yerr=dfTSeries["unc betaFix"], fmt="ys", capsize=3)
    plt.xlabel("Temperature [K]")
    plt.ylabel("beta factor")
    plt.show()

    # dfModeEnergy = pd.read_csv(filePathModeEnergy, sep="\t")
    # temperatureModeEnergy = dfModeEnergy["temperature [K]"]
    # modeEnergy = dfModeEnergy["mode energy [eV]"]
    # uncModeEnergy = dfModeEnergy["uncertainty"]

    # plt.errorbar(temperatureModeEnergy, modeEnergy, yerr=uncModeEnergy, fmt="b.", capsize=3)
    # plt.xlabel("temperature [K]")
    # plt.ylabel("mode energy [eV]")
    # plt.show()

    # dfThreshold = pd.read_csv(filePathThreshold, sep="\t")
    # temperatureThreshold = dfThreshold["temperature [K]"]
    # threshold = dfThreshold["threshold [mW]"]

    # plt.plot(temperatureThreshold, threshold, "b.")
    # plt.ylabel("threshold [mW]")
    # plt.xlabel("temperature [K]")
    # plt.show()

    # dfQfactor = pd.read_csv(filePathQFactor, sep="\t")
    # temperatureQfactor = dfQfactor["temperature [K]"]
    # Qfactor = dfQfactor["Q-factor"]
    # uncQfactor = dfQfactor["uncertainty"]

    # plt.errorbar(temperatureQfactor, Qfactor, yerr=uncQfactor, fmt="b.", capsize=3)
    # plt.ylabel("Q-factor")
    # plt.xlabel("temperature [K]")
    # plt.show()

    # dfBetaBootstrap = pd.read_csv(filePathbetaBootstrap, sep="\t")
    # temperatureBetaBootstrap = dfBetaBootstrap["temperature [K]"]
    # betaBootstrap = dfBetaBootstrap["beta"]
    # uncBetaBootstrap = dfBetaBootstrap["uncertainty"]

    # dfBetaFixedXi = pd.read_csv(filePathbetaFixedXi, sep="\t")
    # # dfBetaFixedXi = dfBetaFixedXi[~dfBetaFixedXi["temperature [K]"].isin([12, 13, 14, 15])]
    # temperatureBetaFixedXi = dfBetaFixedXi["temperature [K]"]
    # betaFixedXi = dfBetaFixedXi["beta"]
    # uncBetaBFixedXi = dfBetaFixedXi["uncertainty"]

    # dfBetaFixedXiNoQEstimate = pd.read_csv(filePathbetaFixedXiNoQEstimate, sep="\t")
    # temperatureBetaFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["temperature [K]"]
    # betaFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["beta"]
    # uncBetaBFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["uncertainty"]

    # plt.errorbar(temperatureBetaBootstrap, betaBootstrap, yerr=uncBetaBootstrap, fmt="b.", capsize=3, label="Bootstrap")
    # plt.errorbar(temperatureBetaFixedXi, betaFixedXi, yerr=uncBetaBFixedXi, fmt="g.", capsize=3, label="fixed xi")
    # plt.errorbar(temperatureBetaFixedXiNoQEstimate, betaFixedXiNoQEstimate, yerr=uncBetaBFixedXiNoQEstimate, fmt="y.", capsize=3, label="fixed xi no Q")
    # plt.ylabel("beta factor")
    # plt.xlabel("temperature [K]")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()