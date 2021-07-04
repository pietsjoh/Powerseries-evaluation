if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    HeadDir = Path(__file__).resolve().parents[2]
    srcDirPath = (HeadDir / "pysrc").resolve()
    sys.path.append(str(srcDirPath))

    import utils.misc as misc

    fsrCsvPath = (HeadDir / "output" / "fsr.csv").resolve()

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

    plt.errorbar(diameterList, meanList, yerr=uncList, fmt="o", markersize=5, capsize=6)
    plt.xlabel("diameter [µm]")
    plt.ylabel("FSR [eV]")
    plt.show()
