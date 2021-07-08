import cProfile
import pstats

import sys
from pathlib import Path
headDir = Path(__file__).parents[2].resolve()
sys.path.append(str(headDir))

from pysrc.powerseries.combine_ps_tool import CombinePowerSeriesTool

with cProfile.Profile() as pr:
    test = CombinePowerSeriesTool()
    test.run()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()