import pytest
import numpy as np

import sys
from pathlib import Path
headDir = Path(__file__).parents[2].resolve()
sys.path.append(str(headDir))

from pysrc.peak_fit.single_peak_fit_base import PeakFitSuper
from pysrc.peak_fit.single_peak_fit_models import *
from pysrc.utils.random_number_gen import RNGenerator
from pysrc.utils.mock_data_gen import mock_data_gen_1_spec

## If I want to test fit_peak(), then every model needs to be tested
## However, I don't know what to test for

seed = None
maxNumPeaks = 10
x = np.linspace(500, 1000, 1000)

y = mock_data_gen_1_spec(x=x, maxNumPeaks=maxNumPeaks, seed=seed)

class TestPeakFitSuper:

    @pytest.mark.xfail(raises=AssertionError, strict=True, reason="only tuples are accepted")
    def test_initialRange_listInput(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = [3, 4]
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    @pytest.mark.xfail(raises=AssertionError, strict=True, reason="only tuples are accepted")
    def test_initialRange_ndarrayInput(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = np.array([3, 4])
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    @pytest.mark.xfail(raises=AssertionError, strict=True, reason="The tuple has to have length 2.")
    def test_initialRange_3tupleInput(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = 1, 2, 3
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    @pytest.mark.xfail(raises=AssertionError, strict=True, reason="only tuples are accepted")
    def test_initialRange_numberInput(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = 1
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    @pytest.mark.xfail(raises=RuntimeError, reason="Only works, when there is a peak at that specific point.")
    def test_initialRange_SameValues(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = 2, 2
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    @pytest.mark.xfail(raises=RuntimeError, strict=True, reason="Initial range is out of bounce.")
    def test_initialRange_outsideRange(self):
        TestObj = PeakFitSuper(x, y)
        initialRange = 10000, 20000
        print(f"initial range: {initialRange}")
        TestObj.get_peak(initialRange=initialRange)

    def test_peak_in_initialRange(self, seed=None):
        TestObj = PeakFitSuper(x, y)
        gen = RNGenerator(seed=seed)
        initialRange = tuple(gen.integers(low=0, high=x.size - 1, size=2))
        print(f"initial range: {initialRange}")
        try:
            TestObj.get_peak(initialRange)
        except RuntimeError:
            print("No peak could be found in the specified initial range.")
        else:
            minIdx = min(initialRange)
            maxIdx = max(initialRange)
            assert minIdx <= TestObj.peak <= maxIdx

    def test_peakHeightEstimate(self, seed=None):
        TestObj = PeakFitSuper(x, y)
        gen = RNGenerator(seed=seed)
        initialRange = tuple(gen.integers(low=0, high=x.size - 1, size=2))
        print(f"initial range: {initialRange}")
        try:
            TestObj.get_peak(initialRange)
        except RuntimeError:
            print("No peak could be found in the specified initial range.")
        else:
            assert y[TestObj.peak] == TestObj.peakHeightEstimate


