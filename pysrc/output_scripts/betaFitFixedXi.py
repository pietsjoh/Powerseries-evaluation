## todo: plot distribution, think about final uncertainty of beta,
def main():
    import numpy as np
    import pandas as pd # type:ignore
    import matplotlib.pyplot as plt
    import scipy.optimize as optimize # type:ignore
    from pathlib import Path
    from functools import partial
    import typing
    np.set_printoptions(suppress=True)
    headDir: Path = Path(__file__).parents[2]
    filePathPowerseries: Path = (headDir / "output" / "powerseries.csv").resolve()
    filePathBetaFit: Path = (headDir / "output" / "beta_fit.csv").resolve()

    number = typing.Union[int, float, np.number]

    ## loading the powerseries data
    dfPowerseries: pd.DataFrame = pd.read_csv(filePathPowerseries, sep="\t")
    inP: np.ndarray = dfPowerseries["in"].to_numpy()
    outP: np.ndarray = dfPowerseries["out"].to_numpy()
    uncOutP: np.ndarray = dfPowerseries["uncOut"].to_numpy()

    ## loading the data from the beta fit, including the values for xi
    dfBetaFit: pd.DataFrame = pd.read_csv(filePathBetaFit, sep="\t")
    xiMin: number = dfBetaFit["xiMin"].to_numpy()[0]
    xiMax: number = dfBetaFit["xiMax"].to_numpy()[0]
    xiEstimateFit: number = dfBetaFit["xiEstimateFit"].to_numpy()[0]
    fitParamsWithXi: np.ndarray = dfBetaFit["fitParamsBeta"].to_numpy()
    uncFitParamsWithXi: np.ndarray = dfBetaFit["uncFitParamsBeta"].to_numpy()

    ## defining the output curve, including xi as a parameter
    def in_out_curve(x, beta, p, a, xi):
        return a / beta * ( p*x / (p*x + 1) * (1 + xi*beta) * (1 + beta*p*x) - (beta**2)*xi*p*x)

    def log_in_out_curve(x, beta, p, a, xi):
        return np.log(in_out_curve(x, beta, p, a, xi))

    ## definitions of initial parameter guesses and boundaries for the fit parameters
    boundsWithoutXi = (0, [1, np.inf, np.inf])
    p0WithoutXi = (0.5, 1, 1)

    ## fitting using the estimated xi from the fit (using the Q-factor from the fit)
    log_in_out_curve_xi_estimate_fit = partial(log_in_out_curve, xi=xiEstimateFit)
    pXiEstimateFit, covXiEstimateFit = optimize.curve_fit(log_in_out_curve_xi_estimate_fit, outP, np.log(inP),
        p0=p0WithoutXi, bounds=boundsWithoutXi)

    ## fitting using different equally distributed values for in the range [xiMin, xiMax]
    ## the final value for beta and its uncertainty is obtained by calculating the mean / std of the distribution
    numSamples: int = 1000
    xiArr: np.ndarray = np.linspace(xiMin, xiMax, numSamples)
    betaArr: np.ndarray = np.empty(numSamples)
    uncBetaArr: np.ndarray = np.empty(numSamples)
    for i, xi in enumerate(xiArr):
        log_in_out_curve_fixed_xi = partial(log_in_out_curve, xi=xi)
        try:
            p, cov = optimize.curve_fit(log_in_out_curve_fixed_xi, outP, np.log(inP),
                p0=p0WithoutXi, bounds=boundsWithoutXi)
        except RuntimeError:
            print(f"fitting did not work for xi={xi}.")
            betaArr[i] = np.nan
            uncBetaArr[i] = np.nan
        else:
            unc: np.ndarray = np.sqrt(np.diag(cov))
            betaArr[i] = p[0]
            uncBetaArr[i] = unc[0]

    meanBeta: number = np.nanmean(betaArr)
    meanUncBeta: number = np.nanmean(uncBetaArr)
    stdBeta: number = np.nanstd(betaArr)

    print("fit with xi as parameter:")
    print(f"parameters:                     {fitParamsWithXi}")
    print(f"uncertainty:                    {uncFitParamsWithXi}")
    print(f"beta factor:                    {fitParamsWithXi[0]:.7f} \u00B1 {uncFitParamsWithXi[0]:.7f}")
    print()
    print("-"*100)
    print()
    print("fit with fixed xi, using the estimated xi (with Q-factor)")
    print(f"parameters:                     {pXiEstimateFit}")
    print(f"uncertainty:                    {np.sqrt(np.diag(covXiEstimateFit))}")
    print(f"beta factor:                    {pXiEstimateFit[0]:.7f} \u00B1 {np.sqrt(np.diag(covXiEstimateFit))[0]:.7f}")
    print()
    print("-"*100)
    print()
    print("fit using multiple values for a fixed xi to sample a distribution")
    print(f"range for xi:                   [{xiMin, xiMax}]")
    print(f"mean beta factor:               {meanBeta}")
    print(f"mean beta factor uncertainty:   {meanUncBeta}")
    print(f"std beta factor:                {stdBeta}")

if __name__ == "__main__":
    main()