def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    headDir = Path(__file__).resolve().parents[2]
    # print(headDir)
    filePathThreshold = (headDir / "output" / "threshold2.csv").resolve()
    filePathbetaBootstrap = (headDir / "output" / "beta_bootstrap.csv").resolve()
    filePathbetaFixedXi = (headDir / "output" / "beta_fixed_xi.csv").resolve()
    filePathbetaFixedXiNoQEstimate = (headDir / "output" / "beta_fixed_xi_no_q_estimate.csv").resolve()
    filePathQFactor = (headDir / "output" / "Q-factor.csv").resolve()


    dfThreshold = pd.read_csv(filePathThreshold, sep="\t")
    diameterThreshold = dfThreshold["diameter [µm]"]
    threshold = dfThreshold["threshold [mW]"]

    plt.plot(diameterThreshold, threshold, "b.")
    plt.ylabel("threshold [mW]")
    plt.xlabel("diameter [µm]")
    plt.show()

    dfQfactor = pd.read_csv(filePathQFactor, sep="\t")
    dfQfactor = dfQfactor[dfQfactor["diameter [µm]"] != 18]
    diameterQfactor = dfQfactor["diameter [µm]"]
    Qfactor = dfQfactor["Q-factor"]
    uncQfactor = dfQfactor["uncertainty"]

    plt.errorbar(diameterQfactor, Qfactor, yerr=uncQfactor, fmt="b.", capsize=3)
    plt.ylabel("Q-factor")
    plt.xlabel("diameter [µm]")
    plt.show()

    dfBetaBootstrap = pd.read_csv(filePathbetaBootstrap, sep="\t")
    diameterBetaBootstrap = dfBetaBootstrap["diameter [µm]"]
    betaBootstrap = dfBetaBootstrap["beta"]
    uncBetaBootstrap = dfBetaBootstrap["uncertainty"]

    dfBetaFixedXi = pd.read_csv(filePathbetaFixedXi, sep="\t")
    # dfBetaFixedXi = dfBetaFixedXi[~dfBetaFixedXi["diameter [µm]"].isin([12, 13, 14, 15])]
    diameterBetaFixedXi = dfBetaFixedXi["diameter [µm]"]
    betaFixedXi = dfBetaFixedXi["beta"]
    uncBetaBFixedXi = dfBetaFixedXi["uncertainty"]

    dfBetaFixedXiNoQEstimate = pd.read_csv(filePathbetaFixedXiNoQEstimate, sep="\t")
    diameterBetaFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["diameter [µm]"]
    betaFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["beta"]
    uncBetaBFixedXiNoQEstimate = dfBetaFixedXiNoQEstimate["uncertainty"]

    plt.errorbar(diameterBetaBootstrap, betaBootstrap, yerr=uncBetaBootstrap, fmt="b.", capsize=3, label="Bootstrap")
    # plt.errorbar(diameterBetaFixedXi, betaFixedXi, yerr=uncBetaBFixedXi, fmt="g.", capsize=3, label="fixed xi")
    # plt.errorbar(diameterBetaFixedXiNoQEstimate, betaFixedXiNoQEstimate, yerr=uncBetaBFixedXiNoQEstimate, fmt="y.", capsize=3, label="fixed xi no Q")
    plt.ylabel("beta factor")
    plt.xlabel("diameter [µm]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()