"""Consists of an abstract base class PeakFitSuper contained in single_peak_fit_base.
single_peak_fit_models contains different fit models which inherit from PeakFitSuper.
Currently, Lorentz, Gauss, Voigt and Pseudo-Voigt are available.

For the fitting process, first scipy.signal.find_peaks is used to find the peak with the largest prominence
in the provided initial range. Then a background fitting mode is applied.
Finally the found peak is fitted (scipy.optimize.curve_fit) using the initial parameters obtained from scipy.signal.
"""