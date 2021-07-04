if __name__ == "__main__":
    import sys
    from pathlib import Path

    HeadDir = Path(__file__).resolve().parents[2]
    srcDirPath = (HeadDir / "pysrc").resolve()
    sys.path.append(str(srcDirPath))

    from FSR_analysis.fsr_selector import FSRselector
    import utils.misc as misc



    diameterInput = input("diameter: ")
    diameter = misc.diameter_decode(diameterInput, returnStr=True)

    dataPath = (HeadDir / "sorted_data" / diameter / "full_spectra" ).resolve()

    files = dataPath.glob("*")
    for file in files:
        test = FSRselector(file)