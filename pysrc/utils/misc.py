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

number = typing.Union[float, int]
listOfNums = list[number]
listOrArray = typing.Union[list, np.ndarray]
intOrNone = typing.Union[int, None]
floatOrNone = typing.Union[float, None]
numberOrNone = typing.Union[number, None]

diameterList: listOfNums = sorted(list(range(1, 21, 1)) + list(np.arange(1.5, 8.5, 1)))
"""list: List of available diameters
"""

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
    assert isinstance(values, (list, np.ndarray))
    n: int = len(values)
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
    std = np.std(values, ddof=1) # type: ignore
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
    maxData: number = max(data)
    minData: number = min(data)
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