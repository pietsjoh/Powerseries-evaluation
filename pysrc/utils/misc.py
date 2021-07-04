import numpy as np
import scipy.stats as stats

import sys
from pathlib import Path
headDirPath = Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))
from setup.config_logging import LoggingConfig
loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

diameterList = sorted(list(range(1, 21, 1)) + list(np.arange(1.5, 8.5, 1)))

def unc_mean(values, intv="1sigma", std=None):
    """
    Berechnet die Unsicherheit eines Mittelwertes für den Array (values) im Intervall (intv).

    values: Werte für die die Unsicherheit des Mittelwertes berechnet werden soll

    intv:   Anteil der Verteilung die in dem Bereich (-intv, intv) liegt.
            Beispiel: intv=0.5 -> symmetrische Grenzen der t-Verteilung, sodass 50% der Werte enthalten sind wird berechnet.
            Ausgegeben wird allerdings nur der positive Wert.
            Neben dem Anteil sind auch die Werte "1sigma", "2sigma" und "3sigma" verfügbar, die an die Normalverteilung angelehnt sind.
            Bei dieser ist es so, dass im Intervall +/- sigma 68.27% der Messwerte zu finden sind.
    """
    assert isinstance(values, (list, np.ndarray))
    n = len(values)
    dof = n - 1
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

    stud_t = stats.t.interval(alpha, dof)[1]
    std = np.std(values, ddof=1)
    unc_mean = stud_t*std/np.sqrt(n)
    return unc_mean

def histo_number_of_bins(data):
    maxData = max(data)
    minData = min(data)
    widthOfBins = 2*stats.iqr(data) / len(data)**(1/3)
    numberOfBins = int(np.ceil(( maxData - minData ) / widthOfBins))
    return numberOfBins

def int_decode(strInt):
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

def float_decode(strFloat):
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

def diameter_decode(strDia, returnStr=False):
    try:
        diameter = int(strDia[:2])
    except ValueError:
        try:
            diameter2 = int(strDia[2])
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
        strDiameter = str(diameter).replace(".", "-")
        return strDiameter
    else:
        return diameter
