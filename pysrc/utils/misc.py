"""Contains miscellaneous functions. Statistics and str to number converter.

Attributes
----------
diameterList: list
    List of available diameters, used to check if diameter input is valid.
"""
import numpy as np
import scipy.stats as stats # type: ignore
import typing

import sys
from pathlib import Path
headDirPath: Path = Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))
from setup.config_logging import LoggingConfig
loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

number = typing.Union[float, int, np.number]
listOfNums = typing.List[number]
listOrArray = typing.Union[list, np.ndarray]
intOrNone = typing.Union[int, None]
floatOrNone = typing.Union[float, None]
numberOrNone = typing.Union[number, None]

diameterList: listOfNums = sorted(list(range(1, 21, 1)) + list(np.arange(1.5, 8.5, 1)))

def input_loop(func: typing.Callable) -> typing.Callable:
    """Decorator that calls the function until it returns false.
    """
    def wrapper(*args, **kwargs):
        j: bool = func(*args, **kwargs)
        while j:
            j = func(*args, **kwargs)
    return wrapper

def weighted_mean(data: np.ndarray, uncertainties: np.ndarray) -> typing.Tuple[np.number, np.number]:
    """Calculates the weighted mean and its uncertainty of the data weighted with the uncertainties.
    """
    weights: np.ndarray = 1 / uncertainties **2
    weightsSum: np.number = np.sum(weights)
    mean: np.number = ( weights @ data ) / weightsSum
    uncertaintyMean: np.number = 1 / np.sqrt(weightsSum)
    return mean, uncertaintyMean

def round_value(value: number, uncertainty: number, useScientific: bool = False, printWarning: bool = False) -> str:
    """Rounds the value and its uncertainty according to DIN 1333.

    Parameters
    ----------
    value: number
        To be rounded value
    uncertainty: number
        The uncertainty of the value
    useScientific: bool
        Whether to use scientific representation for the resulting string or not.
        Keep in mind that even when this is set to false, scientific notation can still occur.
        This happens when there are more than 16 digits in front of the comma or
        if the number is smaller than 0 and there are more than 5 digits behind the comma.
    printWarning: bool
        Whether to print a warning message when the uncertainty exceeds the value

    Returns
    -------
    str:
        rounded value +/- uncertainty

    Example
    -------
    value = 1.567, uncertainty = 0.45

    1.6 \u00B1 0.5 is returned
    """
    assert np.issubdtype(type(value), np.integer) or np.issubdtype(type(value), np.floating)
    assert np.issubdtype(type(uncertainty), np.integer) or np.issubdtype(type(uncertainty), np.floating)
    assert isinstance(useScientific, bool)
    assert isinstance(printWarning, bool)

    ## extract the position in which shall be rounded
    try:
        nonzeroPos: number = int(np.ceil( -np.log10(uncertainty) ))
    except OverflowError:
        print("-"*74)
        print("ERROR: The uncertainty is too small to handle. Returning None.")
        print("-"*74)
        return None
    else:
        ## if the first nonzero digit is 1 or 2, then the digit behind is rounded
        if uncertainty * 10 ** nonzeroPos < 3:
            nonzeroPos += 1

        roundVal: number = np.round(value, nonzeroPos)
        ## uncertainty is always rounded up
        roundUnc: number = np.round(np.ceil(uncertainty * 10 ** nonzeroPos) * 10 ** (-nonzeroPos), nonzeroPos)

        if roundUnc >= roundVal and printWarning:
            print("-"*64)
            print("WARNING: The uncertainty of the values exceeds the value itself.")
            print("-"*64)

        if useScientific:
            ## personal preference of representation ( 10 +/- 4 => 1 +/- 0.4)
            nonzeroPos -= 1
            roundValScientific: number = np.round(roundVal * 10 ** nonzeroPos, 1)
            roundUncScientific: number = np.round(roundUnc * 10 ** nonzeroPos, 1)
            sign: str
            if nonzeroPos <= 0:
                sign = "\u207A"
                nonzeroPos *= -1
            else:
                sign = "\u207B"
            nonzeroPosStr: str = str(nonzeroPos)
            exponentStr: str = sign
            for i in nonzeroPosStr:
                exponentStr += int_to_unicode_superscript(i)
            return f"({roundValScientific} \u00B1 {roundUncScientific})\u00B710{exponentStr}"
        else:
        ## personal preference for the output (removes trailing .0 -> 9.0 +/-3.0 => 9 +/- 3)
            if roundUnc >= 3:
                roundUnc = int(roundUnc)
                roundVal = int(roundVal)
            return f"{roundVal} \u00B1 {roundUnc}"

def int_to_unicode_superscript(number: str) -> str:
    """Takes an integer [0, 9] as a str and transforms it into an unicode superscript string.
    """
    if number == "0":
        return "\u2070"
    elif number == "1":
        return "\u00B9"
    elif number == "2":
        return "\u00B2"
    elif number == "3":
        return "\u00B3"
    elif number == "4":
        return "\u2074"
    elif number == "5":
        return "\u2075"
    elif number == "6":
        return "\u2076"
    elif number == "7":
        return "\u2077"
    elif number == "8":
        return "\u2078"
    elif number == "9":
        return "\u2079"
    else:
        raise ValueError(f"{number} is an invalid input.")

def unc_mean(values: listOrArray, intv: str="1sigma") -> number:
    """
    Calculates the uncertainty for the mean of the values using the student-t-distribution.

    Parameters
    ----------
    values: list, np.ndarray
        The uncertainty is calculated for the mean of these values

    intv: str, int, default="1sigma"
        When int -> proportion of the underlying distribution (-intv, intv) is used.
        For example, if intv=0.5, then the uncertainty is calulated symmetrically from the t-distribution
        such that 50% of the values are inside. In this case intv should be in [0, 1].

        Furthermore, the strings "1sigma", "2sigma" and "3sigma" are accepted as input.
        These use values for intv according to the normal distribution (1sigma= 0.6827).

    Returns
    -------
    float:
        one-sided uncertainty for the mean based on the student-t-distribution

    Raises
    ------
    AssertionError:
        Input has not the correct datatype, or intv is outside of [0, 1]
    """
    # assert isinstance(values, (list, np.ndarray))
    n: int = len(values)
    if n == 1:
        return 0
    dof: int = n - 1
    alpha: number
    if intv == "1sigma":
        alpha = 0.6827
    elif intv == "2sigma":
        alpha = 0.9545
    elif intv == "3sigma":
        alpha = 0.9973
    else:
        assert isinstance(intv, (float, np.floating))
        assert 0 < intv < 1
        alpha = intv

    stud_t: number = stats.t.interval(alpha, dof)[1]
    std: number = np.std(values, ddof=1)
    unc_mean: number = stud_t*std/np.sqrt(n)
    return unc_mean

def histo_number_of_bins(data: listOrArray) -> int:
    """Estimates the number of bins for a histogram after Freedman and Diaconis.

    Parameters
    ----------
    data: list, np.ndarray
        The data for which the number of histogram bins should be calculated.

    Returns
    -------
    int
        number of histogram bins
    """
    assert isinstance(data, (list, np.ndarray))
    maxData: number = np.amax(data)
    minData: number = np.amin(data)
    widthOfBins: number = 2*stats.iqr(data) / len(data)**(1/3)
    numberOfBins: int = int(np.ceil(( maxData - minData ) / widthOfBins))
    return numberOfBins

def int_decode(strInt: str) -> intOrNone:
    """Transforms a string into an integer using int(). Only for ints >= 0.

    Parameters
    ----------
    strInt: str
        To be transformed string

    Returns
    -------
    int:
        When the string could be transformed into an int
    None:
        When the string could not be transformed into an int, or if the value is smaller than 0
    """
    try:
        value = int(strInt)
    except ValueError:
        logger.error(f"ValueError: '{strInt}' is not a valid input (only integers are accepted).")
        return None
    else:
        if value < 0:
            logger.error(f"ValueError: '{value}' is not a valid input (only positive integers are valid).")
            return None
        else:
            return value

def float_decode(strFloat: str) -> floatOrNone:
    """Transforms a string into a float using float(). Only for floats >= 0.

    Parameters
    ----------
    strFloat: str
        To be transformed string

    Returns
    -------
    float:
        When the transformation was successful
    None:
        When the transformation was not successful, or if the float would be smaller than 0

    """
    try:
        value = float(strFloat)
    except ValueError:
        logger.error(f"ValueError: {strFloat} is not a valid input (only numbers are accepted).")
        return None
    else:
        if value < 0:
            logger.error(f"ValueError: '{value}' is not a valid input (only positive numbers are valid).")
            return None
        else:
            return value

def diameter_decode(strDia: str, returnStr: bool=False) -> typing.Union[numberOrNone, str]:
    """Checks whether a string could be transformed into a diameter.

    Here it is checked whether the extracted diameter is part of a specified list (diameterList).

    Parameters
    ----------
    strDia: str
        To be transformed string

    returnStr: bool, default=False
        Whether to return the diameter as a string (True) or as a number (False)

    Returns
    -------
    float/int:
        when resturnStr=False
    str:
        when returnStr=True
    None:
        if no number could be detected or if the diameter is not part of the specified list
    """
    try:
        diameter: number = int(strDia[:2])
    except ValueError:
        try:
            diameter2: number = int(strDia[2])
        except ValueError:
            try:
                diameter = int(strDia[:1])
            except ValueError:
                logger.error(f"ValueError: The diameter string '{strDia}' could not be converted to a number.")
                return None
        except IndexError:
            logger.error(f"IndexError: The diameter string '{strDia}' could not be converted to a number.")
            return None
        else:
            diameter = int(strDia[:1]) + diameter2 / 10
    else:
        diameter = int(strDia[:2])
    if diameter not in diameterList:
        logger.error(f"ValueError: The diameter '{diameter}' is not part of the provided list of possible diameters.")
        return None
    if returnStr:
        strDiameter: str = str(diameter).replace(".", "-")
        return strDiameter
    else:
        return diameter

## just testing
if __name__ == "__main__":
    print(round_value(1981237849178909.5123518239401783491, 0.000123784, useScientific=False))
    # print(int_to_unicode_superscript("1"))