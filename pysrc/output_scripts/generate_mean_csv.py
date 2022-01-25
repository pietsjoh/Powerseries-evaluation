def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import sys

    headDir = Path(__file__).resolve().parents[2]
    srcDir = (headDir / "pysrc").resolve()
    outputDir = (headDir / "output_plots").resolve()
    sys.path.append(str(srcDir))
    import utils.misc as misc

    filePathThresholdAll = (outputDir / "threshold_all.csv").resolve()
    filePathBetaBtAll = (outputDir / "beta_bt_all.csv").resolve()
    filePathBetaFixAll = (outputDir / "beta_fix_all.csv").resolve()
    filePathQfactorAll = (outputDir / "Q-factor_all.csv").resolve()

    filePathThresholdMean = (outputDir / "threshold_mean.csv").resolve()
    filePathQfactorMean = (outputDir / "Q-factor_mean.csv").resolve()
    filePathBetaBtWeightedMean = (outputDir / "beta_bt_weighted_mean.csv").resolve()
    filePathBetaBtMean = (outputDir / "beta_bt_mean.csv").resolve()
    filePathBetaFixWeightedMean = (outputDir / "beta_fix_weighted_mean.csv").resolve()
    filePathBetaFixMean = (outputDir / "beta_fix_mean.csv").resolve()

    thresholdAll = pd.read_csv(filePathThresholdAll, comment="#", sep="\t")
    groupedThreshold = thresholdAll.groupby(["diameter"])
    thresholdMean = groupedThreshold.agg([np.mean, misc.unc_mean])
    thresholdMean.columns = ["mean", "unc"]
    # print(thresholdMean)
    # plt.errorbar(thresholdMean.index, thresholdMean["mean"], yerr=thresholdMean["unc"], fmt="b.")
    # plt.show()
    thresholdMean.to_csv(filePathThresholdMean, sep="\t")

    QfactorAll = pd.read_csv(filePathQfactorAll, comment="#", sep="\t")
    QfactorAll["_weights"] = 1 / QfactorAll["unc"] ** 2
    QfactorAll["_weight_times_data"] = QfactorAll["_weights"] * QfactorAll["Q-factor"]
    groupedQfactor = QfactorAll.groupby(["diameter"])
    QfactorMean = (groupedQfactor["_weight_times_data"].sum() / groupedQfactor["_weights"].sum()).rename("weighted mean").to_frame()
    QfactorMean["unc"] = (1 / np.sqrt(groupedQfactor["_weights"].sum())).rename("unc")
    # print(QfactorMean)
    # plt.errorbar(QfactorMean.index, QfactorMean["weighted mean"], yerr=QfactorMean["unc"], fmt="b.")
    # plt.show()
    QfactorMean.to_csv(filePathQfactorMean, sep="\t")

    BetaBtAll = pd.read_csv(filePathBetaBtAll, comment="#", sep="\t")
    BetaBtAll["_weights"] = 1 / BetaBtAll["unc"] ** 2
    BetaBtAll["_weight_times_data"] = BetaBtAll["_weights"] * BetaBtAll["beta"]
    groupedBetaBt = BetaBtAll.groupby(["diameter"])
    BetaBtWeightedMean = (groupedBetaBt["_weight_times_data"].sum() / groupedBetaBt["_weights"].sum()).rename("weighted mean").to_frame()
    BetaBtWeightedMean["unc"] = (1 / np.sqrt(groupedBetaBt["_weights"].sum())).rename("unc")
    # print(BetaBtMean)
    # plt.errorbar(BetaBtMean.index, BetaBtMean["weighted mean"], yerr=BetaBtMean["unc"], fmt="b.")
    # plt.show()
    BetaBtWeightedMean.to_csv(filePathBetaBtWeightedMean, sep="\t")

    BetaFixAll = pd.read_csv(filePathBetaFixAll, comment="#", sep="\t")
    BetaFixAll["_weights"] = 1 / BetaFixAll["unc"] ** 2
    BetaFixAll["_weight_times_data"] = BetaFixAll["_weights"] * BetaFixAll["beta"]
    groupedBetaFix = BetaFixAll.groupby(["diameter"])
    BetaFixWeightedMean = (groupedBetaFix["_weight_times_data"].sum() / groupedBetaFix["_weights"].sum()).rename("weighted mean").to_frame()
    BetaFixWeightedMean["unc"] = (1 / np.sqrt(groupedBetaFix["_weights"].sum())).rename("unc")
    # print(BetaFixMean)
    # plt.errorbar(BetaFixMean.index, BetaFixMean["weighted mean"], yerr=BetaFixMean["unc"], fmt="b.")
    # plt.show()
    BetaFixWeightedMean.to_csv(filePathBetaFixWeightedMean, sep="\t")

    BetaFixAll = pd.read_csv(filePathBetaFixAll, comment="#", sep="\t")
    BetaFixAll.drop("unc", axis=1, inplace=True)
    groupedBetaFix = BetaFixAll.groupby(["diameter"])
    BetaFixMean = groupedBetaFix.agg([np.mean, misc.unc_mean])
    BetaFixMean.columns = ["mean", "unc"]
    BetaFixMean.to_csv(filePathBetaFixMean, sep="\t")

    BetaBtAll = pd.read_csv(filePathBetaBtAll, comment="#", sep="\t")
    BetaBtAll.drop("unc", axis=1, inplace=True)
    groupedBetaBt = BetaBtAll.groupby(["diameter"])
    BetaBtMean = groupedBetaBt.agg([np.mean, misc.unc_mean])
    BetaBtMean.columns = ["mean", "unc"]
    BetaBtMean.to_csv(filePathBetaBtMean, sep="\t")

if __name__ == "__main__":
    main()