# Currently known Bugs/Problems

> general

- sometimes neither local_left nor local_right background fit works(20210304_NP7509_Ni_8µm_20K_Powerserie_1-01s_deteOD05_fine1AllSpectra.dat,20210304_NP7509_Ni_8µm_20K_Powerserie_1-01s_deteOD2-5_fine2AllSpectra.dat)
- maxInitRange < not for None [x]
- constantPeakWidth has to be an integer [x]
  - I think that it should be that way [x]
- setting constantPeakWidth to 50 might be a problem for small data sets [x]
  - the same is probably true for more parameters (fitRangeScale) [x]
- beta factor fit does not work always
  - initial parameters for fit?
  - or use different fitting routine?
- calc beta and plots both initialize a new instance of PlotPowerSeries
  -future: when beta fit works -> no seperate commands needed???

> eval_ps.py

- crashed once because inputPower exceeded in get_power_dependent_data (maybe because fitting did not work) [x]
  - crashed again like that, this time after setting constantPeakWidth with excluded points [x]
  - do not know why yet, cannot reproduce this error [x]
  - I think this is fixed now [x]
- number of snapshots does not match when excluding points [x]
  - probably fixed now [x]

> combine_ps_tool.py

- check_eval_run(), works not as expected when it has to run ps (--> plots things simultaneously) [x]
- chrashes when typing plot before 'run ps' [x]
- chrashed when injecting a wrong diameter for 'set diameter' [x]

> single_peak_fit_base.py

- image not shown correctly when fitting does not work [x]
  - seems to work now [x]
- ValueError: x must be increasing if s > 0 (problem with ordering from background fitting)
  - file: 20210304_NP7509_Ni_19µm_20K_Powerserie_1-01s_deteOD05_fine1AllSpectra.dat

> plot_ps.py

- constant lines do look off, when the upper S-Shape is the first entry
- if self.inputPower[self.lenInputPowerArr[0] - 1] < self.inputPower[-1]: IndexError: index 40 is out of bounds for axis 0 with size 38

> bootstrap.py

- the following error happened once
- seems like the number of bins is the problem

plt.hist(self.parameterArr, numberOfBins)
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\matplotlib\pyplot.py", line 2685,in hist
  return gca().hist(
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\matplotlib\__init__.py", line 1447,in inner
  return func(ax, *map(sanitize_sequence, args), **kwargs)
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\matplotlib\axes\_axes.py", line6651, in hist
  m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
File "<__array_function__ internals>", line 5, in histogram
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\numpy\lib\histograms.py", line 792,in histogram
  bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\numpy\lib\histograms.py", line 446,in _get_bin_edges
  bin_edges = np.linspace(
File "<__array_function__ internals>", line 5, in linspace
File "C:\Users\Johannes\Documents\Uni\Bachelorarbeit\data-evaluation\env\lib\site-packages\numpy\core\function_base.py", line135, in linspace
  y = _nx.arange(0, num, dtype=dt).reshape((-1,) + (1,) * ndim(delta))
ValueError: Maximum allowed size exceeded
