import cProfile
import pstats

import sys
from pathlib import Path
headDir: Path = Path(__file__).parents[2].resolve()
sys.path.append(str(headDir))

performanceProfileFileLocation: str = str((headDir / "tests" / "performance_tests" / "performance_profile.prof").resolve())

with cProfile.Profile() as pr:
    from pysrc.powerseries.combine_ps_tool import CombinePowerSeriesTool
    test = CombinePowerSeriesTool()
    # test.run()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
stats.dump_stats(filename=performanceProfileFileLocation)