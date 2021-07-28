
def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit as cf
    from pathlib import Path
    np.set_printoptions(suppress=True)
    headDir = Path(__file__).resolve().parent
    filePath = (headDir / "output" / "powerseries.csv").resolve()

    df = pd.read_csv(filePath, sep="\t")
    inP = df["in"].to_numpy()
    outP = df["out"].to_numpy()
    unc = df["uncOut"].to_numpy()

    def in_out_curve_with_xi(x, beta, p, a, xi):
        return a / beta * ( p*x / (p*x + 1) * (1 + xi*beta) * (1 + beta*p*x) - (beta**2)*xi*p*x)

    xiEstimate = 23.70134

    in_out_curve_without_xi = lambda x, beta, p, a : in_out_curve_with_xi(x, beta, p, a, xi=xiEstimate)

    def log_in_out_curve_with_xi(x, *args):
        return np.log(in_out_curve_with_xi(x, *args))

    def log_in_out_curve_without_xi(x, *args):
        return np.log(in_out_curve_without_xi(x, *args))

    boundsWithXi = (0, [1, np.inf, np.inf, np.inf])
    boundsWithoutXi = (0, [1, np.inf, np.inf])
    p0WithXi = (0.05, 1, 1, xiEstimate)
    p0WithoutXi = (0.05, 1, 1)
    weights = unc


    pWithXi, covWithXi = cf(log_in_out_curve_with_xi, outP, np.log(inP), p0=p0WithXi)
    pWithoutXi, covWithoutXi = cf(log_in_out_curve_without_xi, outP, np.log(inP), p0=p0WithoutXi)

    print("fit with xi as parameter:")
    print(f"parameters: {pWithXi}")
    print(f"uncertainty: {np.sqrt(np.diag(covWithXi))}")
    print("-"*100)
    print("fit with fixed xi")
    print(f"parameters: {pWithoutXi}")
    print(f"uncertainty: {np.sqrt(np.diag(covWithoutXi))}")
    print(f"beta factor fixed xi: {pWithoutXi[0]:.7f} \u00B1 {np.sqrt(np.diag(covWithoutXi))[0]:.7f}")

    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax1, ax2 = axs
    outputPlotArr = np.logspace(np.log10(np.amin(outP)), np.log10(np.amax(outP)), 100)
    inputPlotArrWithXi = in_out_curve_with_xi(outputPlotArr, *pWithXi)
    inputPlotArrWithoutXi = in_out_curve_without_xi(outputPlotArr, *pWithoutXi)
    ax1.errorbar(inP, outP, yerr=unc, fmt="b.", capsize=3)
    ax2.errorbar(inP, outP, yerr=unc, fmt="b.", capsize=3)
    ax1.plot(inputPlotArrWithXi, outputPlotArr, color="orange")
    ax1.set_title("fit with xi as parameter")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax2.plot(inputPlotArrWithoutXi, outputPlotArr, color="orange")
    ax2.set_title("fit with fixed xi")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plt.show()

if __name__ == "__main__":
    main()