"""Contains the class FileNameReader, which can be used to extract information from a string.
I will use it to extract the diameter/temperature from the file names.
"""

class FileNameReader:
    """Looks for special sequence in a string.

    The init method is used to define the setup for finding the special sequence
    The call method can then be used to extract the sequence from a filename using
    this setup.

    The following attributes are set upon initialization.

    mandatory: [name, indicator, splitter]

    keyword args: [indicatorAtStart]

    For more information about these arguments take a look at the attributes section.

    Attributes
    ----------
        name: str, set by __init__
            name of the special sequence to look for

        indicator: str, set by __init__
            indicator of the special sequence
            This indicator should not be a part of the sequence.
            Instead it should either be some substring directly before or
            after the desired sequence.

        splitter: str, set by __init__
            Indicator of the other end, can have multiple occurences in the string.
            The string is split at the first possible position.

        indicatorAtStart: bool, set by __init__
            if True, then the looked for sequence is expected directly after the indicator
            if False, then the looked for sequence is expected directly in front of the indicator

    Example
    -------
        filename example: "NP7509_Ni_2µm_20K_Powerserie_1-01s_deteOD0_fine2AllSpectra.dat"

        name: diameter

        result: 2

        Setup 1:
            indicator = _Ni_

            splitter = µm

            indicatorAtStart = True

        Setup 2:
            indicator = µm

            splitter = _

            indicatorAtStart = True
    """
    def __init__(self, name: str, indicator: str, splitter: str,
                indicatorAtStart=True) -> None:
        assert isinstance(name, str)
        assert isinstance(indicator, str)
        assert isinstance(splitter, str)
        assert isinstance(indicatorAtStart, bool)
        self.name: str = name.casefold()
        self.indicator: str = indicator.casefold()
        self.splitter: str = splitter.casefold()
        self.indicatorAtStart: bool = indicatorAtStart

    def __call__(self, fileName: str) -> str:
        """Returns the from a string extracted special sequence.

        Parameters
        ----------
            fileName: str
                name of the file from that the sequence shall be extracted

        Returns
        -------
            str:
                the special sequence

        Raises
        ------
            AssertionError:
                when fileName is not a string,
                when the indicator or the splitter are not part of the fileName
        """
        fileNameCf = fileName.casefold()
        assert isinstance(fileNameCf, str)
        assert self.indicator in fileNameCf
        indicatorIdx: int = fileNameCf.find(self.indicator)
        if self.indicatorAtStart:
            fileNameIndicatorToEnd: str = fileNameCf[indicatorIdx + len(self.indicator) : ]
            assert self.splitter in fileNameIndicatorToEnd
            return fileNameIndicatorToEnd.split(self.splitter)[0]
        else:
            fileNameStartToIndicator: str = fileNameCf[ : indicatorIdx]
            assert self.splitter in fileNameStartToIndicator
            return fileNameStartToIndicator.split(self.splitter)[-1]


if __name__ == "__main__":
    ## just testing
    test = FileNameReader("diameter", "µm", "_", indicatorAtStart=False)
    testFileName = "NP7509_Ni_2µm_20K_Powerserie_1-01s_deteOD0_fine2AllSpectra.dat"
    print(test(testFileName))