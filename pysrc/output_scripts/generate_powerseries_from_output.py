## first find the file to load and use DataQlab2 to load it
## load data into EvalPowerseries then set attributes and call get_power_dependent_data()
## finally plot the results with PlotPowerSeries
## maybe initialize CombinePowerSeriesTool instead
## I think one may want to change thinks after
## maybe do both in seperate scripts or provide the option in this script
def main():
    import numpy as np
    import pandas as pd #type: ignore
    import sys
    from pathlib import Path
    headDirPath: Path = Path(__file__).parents[2]
    srcDirPath: Path = (headDirPath / "pysrc").resolve()
    sys.path.append(str(srcDirPath))

    from powerseries.eval_ps import EvalPowerSeries
    from powerseries.plot_ps import PlotPowerSeries
    from data_tools.data_formats import DataQlab2

    outDirPath: Path = (headDirPath / "output").resolve()
    filePathPowerseries: Path = (outDirPath / "powerseries.csv").resolve()
    filePathBetaFit: Path = (outDirPath / "beta_fit.csv").resolve()
    filePathSettings: Path = (outDirPath / "settings.csv").resolve()

    dfPowerseries: pd.DataFrame = pd.read_csv(filePathPowerseries, sep="\t")
    dfBetaFit: pd.DataFrame = pd.read_csv(filePathBetaFit, sep="\t")
    dfSettings: pd.DataFrame = pd.read_csv(filePathSettings, sep="\t")

    dataPath: Path = (headDirPath / "sorted_data_diameter").resolve()
    dataPathGen = dataPath.rglob("*AllSpectra.dat")

    fileNames: list = dfSettings.columns.values.tolist()[1:]
    powerseriesList: list = []
    for fileName in fileNames:
        ## extract settings from output/settings.csv
        print(dfSettings[fileName])
        fitmodel: str = dfSettings[fileName][0]
        try:
            exclude: list = list(map(int, dfSettings[fileName][1].replace("[", "").replace("]", "").replace(" ", "").split(",")))
        except ValueError:
            exclude: list = []
        fitRangeScale: float = float(dfSettings[fileName][4])
        outputScale: float = float(dfSettings[fileName][5])
        initRangeStr = dfSettings[fileName][6]
        if initRangeStr is np.nan:
            initRange = None
        else:
            initRange = tuple(map(int, initRangeStr.replace("(", "").replace(")", "").replace(" ", "").split(",")))
        try:
            maxInitRange: int = int(dfSettings[fileName][9])
        except ValueError:
            maxInitRange = None
        background: str = dfSettings[fileName][10]
        constantPeakWidth: int = int(dfSettings[fileName][11])
        intCoverage: float = float(dfSettings[fileName][12])
        minEnergy: float = float(dfSettings[fileName][7])
        if minEnergy is np.nan:
            minEnergy = None
        maxEnergy: float = float(dfSettings[fileName][8])
        if maxEnergy is np.nan:
            maxEnergy = None
        minInputPower: float = float(dfSettings[fileName][2])
        if minInputPower is np.nan:
            minInputPower = None
        maxInputPower: float = float(dfSettings[fileName][3])
        if maxInputPower is np.nan:
            maxInputPower = None

        ## find file with the original data
        for file in dataPathGen:
            if file.name == fileName:
                DataObj = DataQlab2(file)
                Powerseries = EvalPowerSeries(DataObj)
                break
        else:
            print("-"*45)
            print("Warning: file not found in the data directory")
            print("-"*45)
            sys.exit()

        ## set attributes from settings.csv to the Powerseries object
        Powerseries.initRange = initRange
        Powerseries.maxInitRange = maxInitRange
        Powerseries.backgroundFitMode = background
        Powerseries.constantPeakWidth = constantPeakWidth
        Powerseries.intCoverage = intCoverage
        Powerseries.exclude = exclude
        Powerseries.check_input_fitmodel(fitmodel)
        Powerseries.powerScale = outputScale
        Powerseries.fitRangeScale = fitRangeScale
        Powerseries.snapshots = 0
        Powerseries.minInitRangeEnergy = minEnergy
        Powerseries.maxInitRangeEnergy = maxEnergy
        Powerseries.minInputPowerRange = minInputPower
        Powerseries.maxInputPowerRange = maxInputPower
        Powerseries.get_power_dependent_data()

        powerseriesList.append(Powerseries)

    ## extract settings from output/beta_fit.csv
    try:
        bootstrapSeed = int(dfBetaFit["bootstrap seed"][0])
    except KeyError:
        PlotPowerSeriesObj = PlotPowerSeries(powerseriesList)
        PlotPowerSeriesObj.useBootstrap = False
    else:
        PlotPowerSeriesObj = PlotPowerSeries(powerseriesList, bootstrapSeed=bootstrapSeed)
        PlotPowerSeriesObj.useBootstrap = True
    PlotPowerSeriesObj.beta_factor_2()
    PlotPowerSeriesObj.plot_lw_s()
if __name__ == "__main__":
    main()