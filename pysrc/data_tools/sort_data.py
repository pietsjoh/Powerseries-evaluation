if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    import pandas as pd


    diameterIndicator = "NP7509_NI_"
    fineBorder = 30 ##nm

    HeadDirPath = Path(__file__).resolve().parents[2]
    srcDirPath = (HeadDirPath / "pysrc").resolve()
    sys.path.append(str(srcDirPath))
    import utils.console_commands as console
    import utils.misc as misc

    dataDirPath = (HeadDirPath / "data").resolve()

    sortDataDirPath = (HeadDirPath / "sorted_data").resolve()

    spectraPaths = list(dataDirPath.rglob("*AllSpectra.dat"))

    ## sort QLab-2 data only (Allspectra.dat)
    ## first: sort data by diameter

    if not sortDataDirPath.exists():
        sortDataDirPath.mkdir()

    for path in spectraPaths:
        strFileName = str(path)
        # print(strFileName)
        if "tesst" in strFileName or "fail" in strFileName:
            continue
        strFileNameUpper = strFileName.upper()
        fileNameIndex = strFileNameUpper.find(diameterIndicator.upper())
        diameterIndex = fileNameIndex + len(diameterIndicator)
        truncatedStrFileName = strFileName[diameterIndex : ]
        diameter = misc.diameter_decode(truncatedStrFileName, returnStr=True)

        diameterDirPath = (sortDataDirPath / diameter).resolve()
        if not diameterDirPath.exists():
            diameterDirPath.mkdir()
        strParentDir = str(path.parent).split("/")[-1]
        oldFileName = strFileName[fileNameIndex : ]
        # print(oldFileName)
        newFileName = strParentDir + "_" + strFileName[fileNameIndex : ]
        # print(newFileName)
        # print()
        newFilePath = (diameterDirPath / newFileName).resolve()
        if newFilePath.exists():
            continue
        else:
            console.cp(strFileName, str(diameterDirPath))
            console.mv(str((diameterDirPath / oldFileName).resolve()), str(newFilePath))

    ## second sort data by full_spectra and fine_spectra

    diameterDirs = os.listdir(str(sortDataDirPath))
    for diameter in diameterDirs:
        diameterPath = (sortDataDirPath / diameter).resolve()
        files = os.listdir(str(diameterPath))
        fineDataPath = (diameterPath / "fine_spectra").resolve()
        fullDataPath = (diameterPath / "full_spectra").resolve()
        for file in files:
            if file == "fine_spectra" or file == "full_spectra":
                continue
            fileName = file.split("/")[-1]
            print(fileName)
            filePath = (diameterPath / fileName).resolve()
            rowsToSkip = [1, 3]
            data = pd.read_csv(filePath, sep="\t", skiprows=lambda x: x in rowsToSkip)
            wavelengths = data["Wavelength"].to_numpy()[1:]
            wavelengthRange = abs(wavelengths[-1] - wavelengths[0])
            if wavelengthRange <= fineBorder:
                if not fineDataPath.exists():
                    fineDataPath.mkdir()
                console.mv(str(filePath), str((fineDataPath / fileName).resolve()))
            else:
                if not fullDataPath.exists():
                    fullDataPath.mkdir()
                console.mv(str(filePath), str((fullDataPath / fileName).resolve()))

