"""This file contains code for use with "Think Stats", 3rd edition
by Allen B. Downey, available from greenteapress.com

Copyright 2024 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

"""

import bisect
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import comb, factorial

from empiricaldist import FreqTab, Pmf, Cdf, Hazard

from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.integrate import simpson

from IPython.display import display, HTML
from statsmodels.iolib.table import SimpleTable

from IPython.core.magic import register_cell_magic
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
)

# Check if we're running in a Jupyter environment
try:
    get_ipython()
    IN_JUPYTER = True
except NameError:
    IN_JUPYTER = False

def extract_function_name(text):
    """Find a function definition and return its name.

    Args:
        text: string containing function definition.

    Returns:
        string or None: Function name if found, None otherwise.
    """
    pattern = r"def\s+(\w+)\s*\("
    match = re.search(pattern, text)
    if match:
        func_name = match.group(1)
        return func_name
    else:
        return None


def add_method_to(args, cell):
    """Add a method to a class.

    Args:
        args: string containing the class name.
        cell: string containing the function definition.

    Returns:
        string: Status message indicating success or failure.
    """
    # get the name of the function defined in this cell
    func_name = extract_function_name(cell)
    if func_name is None:
        return f"This cell doesn't define any new functions."

    # get the class we're adding it to
    namespace = get_ipython().user_ns
    class_name = args
    cls = namespace.get(class_name, None)
    if cls is None:
        return f"Class '{class_name}' not found."

    # save the old version of the function if it was already defined
    old_func = namespace.get(func_name, None)
    if old_func is not None:
        del namespace[func_name]

    # Execute the cell to define the function
    get_ipython().run_cell(cell)

    # get the newly defined function
    new_func = namespace.get(func_name, None)
    if new_func is None:
        return f"This cell didn't define {func_name}."

    # add the function to the class and remove it from the namespace
    setattr(cls, func_name, new_func)
    del namespace[func_name]

    # restore the old function to the namespace
    if old_func is not None:
        namespace[func_name] = old_func


def expect_error(line, cell):
    """Execute a cell and display the traceback if it raises an exception.

    Args:
        line: Unused.
        cell: string containing code to execute.
    """
    try:
        get_ipython().run_cell(cell)
    except Exception as e:
        get_ipython().run_cell("%tb")


@magic_arguments()
@argument("exception", help="Type of exception to catch")
def expect(line, cell):
    """Execute a cell and display the traceback if it raises the expected exception.

    Args:
        line: string containing the expected exception type.
        cell: string containing code to execute.
    """
    args = parse_argstring(expect, line)
    exception = eval(args.exception)
    try:
        get_ipython().run_cell(cell)
    except exception as e:
        get_ipython().run_cell("%tb")


# Only register cell magics if we're in a Jupyter environment
if IN_JUPYTER:
    register_cell_magic(add_method_to)
    register_cell_magic(expect_error)
    register_cell_magic(expect)

# Make the figures smaller to save some screen real estate.
# The figures generated for the book have DPI 300, so scaling
# them by a factor of 4 restores them to the size in the notebooks.
plt.rcParams["figure.dpi"] = 75
plt.rcParams["figure.figsize"] = [6, 3.5]


def remove_spines():
    """Remove the spines of a plot but keep the ticks visible."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ensure ticks stay visible
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def value_counts(seq, **options):
    """Counts the values in a series and returns sorted.

    Args:
        seq: sequence to count values from.
        **options: passed to pd.Series.value_counts.

    Returns:
        pd.Series: Sorted value counts.
    """
    options = underride(options, dropna=False)
    return pd.Series(seq).value_counts(**options).sort_index()


## Chapter 1

# read_stata is in nsfg.py

def show_table(d):
    """Show a table in a Jupyter notebook.

    Args:
        d: dictionary to show.

    Returns:
        HTML: HTML representation of the table.
    """
    df = pd.DataFrame(d)
    return HTML(df.to_html(index=False))

## Chapter 2

def smallest(ftab, n=10):
    """Returns the smallest n values from a FreqTab.

    Args:
        ftab: FreqTab object to get smallest values from.
        n: int number of values to return.

    Returns:
        list: List of n smallest values.
    """
    return ftab[:n]


def largest(ftab, n=10):
    """Returns the largest n values from a FreqTab.

    Args:
        ftab: FreqTab object to get largest values from.
        n: int number of values to return.

    Returns:
        list: List of n largest values.
    """
    return ftab[-n:]


def two_bar_plots(dist1, dist2, width=0.45, xlabel="", **options):
    """Makes two back-to-back bar plots.

    Args:
        dist1: FreqTab or Pmf object for the left bars.
        dist2: FreqTab or Pmf object for the right bars.
        width: float width of the bars.
        xlabel: string label for the x-axis.
        **options: passed along to plt.bar.
    """
    dist1.bar(align="edge", width=-width, **options)
    underride(options, alpha=0.5)
    dist2.bar(align="edge", width=width, **options)
    decorate(xlabel=xlabel)


def cohen_effect_size(group1, group2):
    """Computes Cohen's effect size for two groups.

    Args:
        group1: sequence containing first group's data.
        group2: sequence containing second group's data.

    Returns:
        float: Cohen's effect size.
    """
    diff = group1.mean() - group2.mean()

    v1, v2 = group1.var(), group2.var()
    n1, n2 = group1.count(), group2.count()
    pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2)

    return diff / np.sqrt(pooled_var)


## Chapter 3

def bias(pmf, name):
    """Pmf of a length-biased sample.

    Args:
        pmf: Pmf object to bias.
        name: string name for the biased Pmf.

    Returns:
        Pmf: biased Pmf.
    """
    # multiply each probability by class size
    ps = pmf.ps * pmf.qs

    # make a new Pmf and normalize it
    new_pmf = Pmf(ps, pmf.qs, name=name)
    new_pmf.normalize()
    return new_pmf

def unbias(pmf, name):
    """Unbias a Pmf by class size.

    Args:
        pmf: Pmf object to unbias.
        name: string name for the unb.

    Returns:
        Pmf: unbiassed Pmf.
    """
    # divide each probability by class size
    ps = pmf.ps / pmf.qs

    new_pmf = Pmf(ps, pmf.qs, name=name)
    new_pmf.normalize()
    return new_pmf

## Chapter 4

def percentile_rank(x, seq):
    """Percentile rank of x.

    Args:
        x: value to find the percentile rank of.
        seq: sequence of values to compare to.

    Returns:
        float: Percentile rank of x.
    """
    return (seq <= x).mean() * 100

def percentile(p, seq):
    """Percentile of a sequence.

    Args:
        p: float percentile to compute (0-100).
        seq: sequence of values to compute the percentile of.

    Returns:
        float: Value at the given percentile.

    Raises:
        ValueError: If p is not between 0 and 100.
    """
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    n = len(seq)
    i = (1 - p / 100) * (n + 1)
    return seq[round(i)]

def skewness(seq):
    """Compute the skewness of a sequence

    Args:
        seq: sequence of numbers to compute the skewness of.

    Returns:
        float: Skewness of the sequence.
    """
    deviations = seq - seq.mean()
    return np.mean(deviations**3) / seq.std(ddof=0) ** 3

def median(cdf):
    """Median of a CDF.

    Args:
        cdf: Cdf object to compute the median of.

    Returns:
        float: Median of the CDF.
    """
    m = cdf.inverse(0.5)
    return m

def iqr(cdf):
    """Interquartile range of a CDF.

    Args:
        cdf: Cdf object to compute the IQR of.

    Returns:
        float: Interquartile range of the CDF.
    """
    low, high = cdf.inverse([0.25, 0.75])
    return high - low

def quartile_skewness(cdf):
    """Quartile skewness of a CDF.

    Args:
        cdf: Cdf object to compute the quartile skewness of.

    Returns:
        float: Quartile skewness of the CDF.
    """
    low, median, high = cdf.inverse([0.25, 0.5, 0.75])
    midpoint = (high + low) / 2
    semi_iqr = (high - low) / 2
    return (midpoint - median) / semi_iqr

def sample_from_cdf(cdf, n):
    """Sample from a CDF.

    Args:
        cdf: Cdf object to sample from.
        n: int number of samples to draw.

    Returns:
        ndarray: Random sample from the CDF.
    """
    ps = np.random.random(size=n)
    return cdf.inverse(ps)

## Chapter 5

def flip(n, p):
    """Flip a coin n times with probability p of heads.

    Args:
        n: int number of flips.
        p: float probability of heads.

    Returns:
        ndarray: Random sample of heads and tails.
    """
    choices = [1, 0]
    probs = [p, 1 - p]
    return np.random.choice(choices, n, p=probs)

def simulate_round(n, p):
    """Simulate a round of n coin flips with probability p of heads.

    Args:
        n: int number of flips.
        p: float probability of heads.

    Returns:
        int: Number of heads in the round.
    """
    seq = flip(n, p)
    return seq.sum()
    

def binomial_pmf(k, n, p):
    """Compute the binomial PMF.

    Args:
        k: int or array-like number of successes.
        n: int number of trials.
        p: float probability of success on a single trial.

    Returns:
        float or ndarray: Probability mass for k successes.
    """
    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))

def simulate_goals(n, p):
    """Simulate the number of goals scored in n soccer games with probability p of scoring a goal.

    Args:
        n: int number of games.
        p: float probability of scoring a goal.

    Returns:
        int: Number of goals scored in n games. 
    """
    return flip(n, p).sum()

def simulate_first_goal(n, p):
    """Simulate the first goal scored in n soccer games with probability p of scoring a goal.

    Args:
        n: int number of games.
        p: float probability of scoring a goal.

    Returns:    
        int: Number of games until the first goal is scored.
    """
    return flip(n, p).argmax()

def poisson_pmf(k, lam):
    """Compute the Poisson PMF.

    Args:
        k: int or array-like number of occurrences.
        lam: float rate parameter (λ) of the Poisson distribution.

    Returns:
        float or ndarray: Probability mass for k occurrences.
    """
    return (lam**k) * np.exp(-lam) / factorial(k)

def exponential_cdf(x, lam):
    """Compute the exponential CDF.

    Args:
        x: float or sequence of floats.
        lam: float rate parameter.

    Returns:
        float or ndarray: Cumulative probability.
    """
    return 1 - np.exp(-lam * x)

def simulate_growth(n):
    """Simulate the growth of a stock over n days.

    Args:
        n: int number of days.

    Returns:
        int: Total growth over n days.
    """
    choices = [1, 2, 3]
    gains = np.random.choice(choices, n)
    return gains.sum()



def make_normal_model(data):
    """Make the Cdf of a normal distribution based on data.

    Args:
        data: sequence of numbers.

    Returns:
        Cdf: Normal distribution model.
    """
    m, s = np.mean(data), np.std(data)
    low, high = m - 4 * s, m + 4 * s
    qs = np.linspace(low, high, 201)
    ps = norm.cdf(qs, m, s)
    return Cdf(ps, qs, name="normal model")

def two_cdf_plots(cdf_model, cdf_data, xlabel="", **options):
    """Plot an empirical CDF and a theoretical model.

    Args:
        cdf_model: Cdf object representing the theoretical model.
        cdf_data: Cdf object representing the empirical data.
        xlabel: string label for the x-axis.
        **options: Control the way cdf_data is plotted.
    """
    cdf_model.plot(ls=":", color="gray")
    cdf_data.plot(**options)
    decorate(xlabel=xlabel, ylabel="CDF")

def simulate_proportionate_growth(n):
    """Simulate proportionate growth over n days.

    Args:
        n: int number of days.

    Returns:
        int: Total growth over n days.
    """
    choices = [1.03, 1.05, 1.07]
    gains = np.random.choice(choices, n)
    return gains.prod()


def read_brfss(filename="CDBRFS08.ASC.gz", compression="gzip", nrows=None):
    """Reads the BRFSS data.

    Args:
        filename: string path to the data file.
        compression: string indicating compression type.
        nrows: optional int number of rows to read, or None for all.

    Returns:
        DataFrame: BRFSS data with cleaned variables.
    """
    # column names and column specs from
    # https://www.cdc.gov/brfss/annual_data/2008/varLayout_table_08.htm
    var_info = [
        ("age", 100, 102, int),
        ("sex", 142, 143, int),
        ("wtyrago", 126, 130, int),
        ("finalwt", 798, 808, int),
        ("wtkg2", 1253, 1258, int),
        ("htm3", 1250, 1253, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)

    colspecs = variables[["start", "end"]].values.tolist()
    names = variables["name"].tolist()

    df = pd.read_fwf(
        filename,
        colspecs=colspecs,
        names=names,
        compression=compression,
        nrows=nrows,
    )

    clean_brfss(df)
    return df


def clean_brfss(df):
    """Recodes BRFSS variables.

    Args:
        df: DataFrame containing BRFSS data to clean.
    """
    df["age"] = df["age"].replace([7, 9], np.nan)
    df["htm3"] = df["htm3"].replace([999], np.nan)
    df["wtkg2"] = df["wtkg2"].replace([99999], np.nan) / 100
    df["wtyrago"] = df.wtyrago.replace([7777, 9999], np.nan)
    df["wtyrago"] = df.wtyrago.apply(
        lambda x: x / 2.2 if x < 9000 else x - 9000
    )


# Chapter 6


def normal_pdf(xs, mu, sigma):
    """Evaluates the normal probability density function.

    Args:
        xs: float or sequence of floats.
        mu: float mean of the distribution.
        sigma: float standard deviation of the distribution.

    Returns:
        float or ndarray: Probability density.
    """
    z = (xs - mu) / sigma
    return np.exp(-(z**2) / 2) / sigma / np.sqrt(2 * np.pi)


class Density:
    """Represents a continuous PDF or CDF."""

    def __init__(self, density_func, domain, name=""):
        """Initializes the Pdf.

        Args:
            density_func: function that computes the density.
            domain: tuple of (low, high) values.
            name: string name for the distribution.
        """
        self.name = name
        self.density_func = density_func
        self.domain = domain

    def __repr__(self):
        """Returns a string representation."""
        return f"Density({self.density_func.__name__}, {self.domain}, name={self.name})"

    def __call__(self, qs):
        """Evaluates this Density at qs.

        Args:
            qs: float or sequence of floats.

        Returns:
            float or ndarray: Probability density.
        """
        return self.density_func(qs)

    def plot(self, qs=None, **options):
        """Plots this Density.

        Args:
            qs: optional ndarray of quantities where the density_func should be evaluated.
            **options: passed along to plt.plot.
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)

        ps = self(qs)
        underride(options, label=self.name)
        plt.plot(qs, ps, **options)


class Pdf(Density):
    """Represents a PDF."""

    def make_pmf(self, qs=None, **options):
        """Makes a discrete approximation to the Pdf.

        Args:
            qs: optional ndarray of quantities where the Pdf should be evaluated.
            **options: passed along to the Pmf constructor.

        Returns:
            Pmf: Discrete approximation of the Pdf.
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)
        ps = self(qs)

        underride(options, name=self.name)
        pmf = Pmf(ps, qs, **options)
        pmf.normalize()
        return pmf


def area_under(pdf, low, high):
    """Find the area under a PDF.

    Args:
        pdf: Pdf object to integrate.
        low: float low end of the interval.
        high: float high end of the interval.

    Returns:
        float: Area under the PDF between low and high.
    """
    qs = np.linspace(low, high, 501)
    ps = pdf(qs)
    return simpson(y=ps, x=qs)


class ContinuousCdf(Density):
    """Represents a CDF."""

    def make_cdf(self, qs=None, **options):
        """Makes a discrete approximation to the CDF.

        Args:
            qs: optional ndarray of quantities where the CDF should be evaluated.
            **options: passed along to the Cdf constructor.

        Returns:
            Cdf: Discrete approximation of the CDF.
        """
        if qs is None:
            low, high = self.domain
            qs = np.linspace(low, high, 201)

        ps = self(qs)

        underride(options, name=self.name)
        cdf = Cdf(ps, qs, **options)
        return cdf


class NormalPdf(Pdf):
    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, domain=None, name=""):
        """Constructs a NormalPdf with given mu and sigma.

        Args:
            mu: float mean of the distribution.
            sigma: float standard deviation of the distribution.
            domain: optional tuple of (low, high) values.
            name: string name for the distribution.
        """
        self.mu = mu
        self.sigma = sigma
        if domain is None:
            domain = mu - 4 * sigma, mu + 4 * sigma
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return (
            f"NormalPdf({self.mu}, {self.sigma}, name='{self.name}')"
        )

    def __call__(self, qs):
        """Evaluates this PDF at qs.

        Args:
            qs: float or sequence of floats.

        Returns:
            float or ndarray: Probability density.
        """
        return normal_pdf(qs, self.mu, self.sigma)


class NormalCdf(ContinuousCdf):
    """Represents the CDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, domain=None, name=""):
        """Constructs a NormalCdf with given mu and sigma.

        Args:
            mu: float mean of the distribution.
            sigma: float standard deviation of the distribution.
            domain: optional tuple of (low, high) values.
            name: string name for the distribution.
        """
        self.mu = mu
        self.sigma = sigma
        if domain is None:
            domain = mu - 4 * sigma, mu + 4 * sigma
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return (
            f"NormalCdf({self.mu}, {self.sigma}, name='{self.name}')"
        )

    def __call__(self, qs):
        """Evaluates this CDF at qs.

        Args:
            qs: float or sequence of floats.

        Returns:
            float or ndarray: Cumulative probability.
        """
        return norm.cdf(qs, self.mu, self.sigma)


def exponential_pdf(x, lam):
    """Evaluates the exponential PDF.

    Args:
        x: float or sequence of floats.
        lam: float rate parameter.

    Returns:
        float or ndarray: Probability density.
    """
    return lam * np.exp(-lam * x)


class ExponentialPdf(Pdf):
    """Represents the PDF of an exponential distribution."""

    def __init__(self, lam=1, domain=None, name=""):
        """Constructs an ExponentialPdf with given lambda.

        Args:
            lam: float rate parameter.
            domain: optional tuple of (low, high) values.
            name: string name for the distribution.
        """
        self.lam = lam
        if domain is None:
            domain = 0, 5.0 / lam
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"ExponentialPdf({self.lam}, name='{self.name}')"

    def __call__(self, qs):
        """Evaluates this PDF at qs.

        Args:
            qs: float or sequence of floats.

        Returns:
            float or ndarray: Probability density.
        """
        return exponential_pdf(qs, self.lam)


class ExponentialCdf(ContinuousCdf):
    """Represents the CDF of an exponential distribution."""

    def __init__(self, lam=1, domain=None, name=""):
        """Constructs an ExponentialCdf with given lambda.

        Args:
            lam: float rate parameter.
            domain: optional tuple of (low, high) values.
            name: string name for the distribution.
        """
        self.lam = lam
        if domain is None:
            domain = 0, 5.0 / lam
        self.domain = domain
        self.name = name

    def __repr__(self):
        """Returns a string representation."""
        return f"ExponentialCdf({self.lam}, name='{self.name}')"

    def __call__(self, qs):
        """Evaluates this CDF at qs.

        Args:
            qs: float or sequence of floats.

        Returns:
            float or ndarray: Cumulative probability.
        """
        return exponential_cdf(qs, self.lam)


def read_baby_boom(filename="babyboom.dat"):
    """Reads the babyboom data.

    Args:
        filename: string path to the data file.

    Returns:
        DataFrame: Baby boom data with time, sex, weight, and minutes columns.
    """
    colspecs = [(1, 8), (9, 16), (17, 24), (25, 32)]
    column_names = ["time", "sex", "weight_g", "minutes"]
    df = pd.read_fwf(
        filename, colspecs=colspecs, names=column_names, skiprows=59
    )
    return df


## Chapter 7


def jitter(seq, std=1):
    """Jitters the values by adding random Gaussian noise.

    Args:
        seq: sequence of numbers to jitter.
        std: float standard deviation of the added noise.

    Returns:
        ndarray: New array with jittered values.
    """
    n = len(seq)
    return np.random.normal(0, std, n) + seq


def standardize(xs):
    """Standardizes a sequence of numbers to have mean 0 and standard deviation 1.
    
    Uses population standard deviation (ddof=0) for consistency with the
    definition of z-scores.

    Args:
        xs: sequence of numbers to standardize.

    Returns:
        ndarray: Standardized values with mean 0 and standard deviation 1.
    """
    return (xs - np.mean(xs)) / np.std(xs)


## Chapter 8


def plot_kde(sample, name="", **options):
    """Plot an estimated PDF using Gaussian Kernel Density Estimation.

    Args:
        sample: sequence of values to estimate PDF from.
        name: string name for the plot.
        **options: passed along to plt.plot.
    """
    kde = gaussian_kde(sample)
    m, s = np.mean(sample), np.std(sample)
    plt.axvline(m, color="gray", ls=":")

    domain = m - 4 * s, m + 4 * s
    pdf = Pdf(kde, domain, name)
    pdf.plot(**options)


## Chapter 9


def make_pmf(sample, low, high):
    """Make a PMF based on KDE.

    Args:
        sample: sequence of values to estimate PMF from.
        low: float low end of the range.
        high: float high end of the range.

    Returns:
        Pmf: Probability mass function based on KDE.
    """
    kde = gaussian_kde(sample)
    qs = np.linspace(low, high, 201)
    ps = kde(qs)
    return Pmf(ps, qs)


## Chapter 10


def display_summary(result):
    """Prints summary statistics from a regression model.

    Args:
        result: RegressionResults object to summarize.
    """
    params = result.summary().tables[1]
    display(params)

    if hasattr(result, "rsquared"):
        row = ["R-squared:", f"{result.rsquared:0.4}"]
    elif hasattr(result, "prsquared"):
        row = ["Pseudo R-squared:", f"{result.prsquared:0.4}"]
    else:
        return
    table = SimpleTable([row])
    display(table)


## Chapter 13


def estimate_hazard(complete, ongoing):
    """Estimates the hazard function.

    Args:
        complete: sequence of complete lifetimes.
        ongoing: sequence of ongoing lifetimes.

    Returns:
        Hazard: Estimated hazard function.
    """
    ft_complete = FreqTab.from_seq(complete)
    ft_ongoing = FreqTab.from_seq(ongoing)

    surv_complete = ft_complete.make_surv()
    surv_ongoing = ft_ongoing.make_surv()

    ts = pd.Index.union(ft_complete.index, ft_ongoing.index)

    at_risk = (
        ft_complete(ts)
        + ft_ongoing(ts)
        + surv_complete(ts)
        + surv_ongoing(ts)
    )

    hs = ft_complete(ts) / at_risk

    return Hazard(hs, ts)


## unassigned


def scatter(df, var1, var2, jitter_std=None, **options):
    """Make a scatter plot and return the coefficient of correlation.

    Args:
        df: DataFrame containing the variables.
        var1: string name of first variable.
        var2: string name of second variable.
        jitter_std: optional float standard deviation of noise to add.
        **options: passed along to plt.scatter.
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1]
    ys = valid[var2]

    if jitter_std is not None:
        xs = jitter(xs, jitter_std)
        ys = jitter(ys, jitter_std)

    underride(options, s=5, alpha=0.2)
    plt.scatter(xs, ys, **options)


def decile_plot(df, var1, var2, **options):
    """Make a decile plot.

    Args:
        df: DataFrame containing the variables.
        var1: string name of first variable.
        var2: string name of second variable.
        **options: passed along to plt.plot.
    """
    valid = df.dropna(subset=[var1, var2])
    deciles = pd.qcut(valid[var1], 10, labels=False)
    df_groupby = valid.groupby(deciles)
    series_groupby = df_groupby[var2]

    low = series_groupby.quantile(0.1)
    median = series_groupby.quantile(0.5)
    high = series_groupby.quantile(0.9)

    xs = df_groupby[var1].median()

    plt.fill_between(xs, low, high, alpha=0.2)
    underride(options, color="C0", label="median")
    plt.plot(xs, median, **options)


def corrcoef(df, var1, var2):
    """Computes the correlation matrix for two variables.

    Args:
        df: DataFrame containing the variables.
        var1: string name of first variable.
        var2: string name of second variable.

    Returns:
        float: Correlation coefficient between var1 and var2.
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1]
    ys = valid[var2]
    return np.corrcoef(xs, ys)[0, 1]


def rankcorr(df, var1, var2):
    """Computes the Spearman rank correlation for two variables.

    Args:
        df: DataFrame containing the variables.
        var1: string name of first variable.
        var2: string name of second variable.

    Returns:
        float: Spearman rank correlation coefficient.
    """
    valid = df.dropna(subset=[var1, var2])
    xs = valid[var1].rank()
    ys = valid[var2].rank()
    return np.corrcoef(xs, ys)[0, 1]


def make_correlated_scatter(xs, ys, rho, **options):
    """Makes a scatter plot with given correlation.

    Args:
        xs: sequence of x values.
        ys: sequence of y values.
        rho: float target correlation coefficient.
        **options: passed along to plt.scatter.
    """
    ys = rho * xs + np.sqrt(1 - rho**2) * ys

    underride(options, s=5, alpha=0.5)
    plt.scatter(xs, ys, **options)
    add_rho(rho)
    remove_spines()


def add_rho(rho):
    """Adds a label to a figure to indicate the correlation.

    Args:
        rho: float correlation coefficient to display.
    """
    ax = plt.gca()
    plt.text(
        0.5,
        0.05,
        f"ρ = {rho}",
        fontsize="x-large",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def make_nonlinear_scatter(xs, ys, kind="quadratic", **options):
    """Makes a scatter plot with a nonlinear relationship.

    Args:
        xs: sequence of x values.
        ys: sequence of y values.
        kind: string type of nonlinear relationship. One of:
            - 'quadratic': adds x^2 to y
            - 'sinusoid': adds 10*sin(3x) to y
            - 'abs': adds -|x| to y
        **options: passed along to plt.scatter.

    Returns:
        float: Correlation coefficient of the resulting scatter plot.
    """
    if kind == "quadratic":
        ys = ys + xs**2
    elif kind == "sinusoid":
        ys = ys + 10 * np.sin(3 * xs)
    elif kind == "abs":
        ys = ys / 4 - np.abs(xs)

    underride(options, s=5, alpha=0.5)
    plt.scatter(xs, ys, **options)
    remove_spines()
    r = np.corrcoef(xs, ys)[0, 1]
    return r


def remove_spines():
    """Remove the spines from a plot."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks([])


def cov(xs, ys):
    """Covariance of two variables.

    Args:
        xs: sequence of values for first variable.
        ys: sequence of values for second variable.

    Returns:
        float: Covariance between xs and ys.
    """
    xbar = np.mean(xs)
    ybar = np.mean(ys)
    dx = xs - xbar
    dy = ys - ybar
    cov = np.mean(dx * dy)
    return cov


def corr(xs, ys):
    """Correlation coefficient for two variables.

    Args:
        xs: sequence of values for first variable.
        ys: sequence of values for second variable.

    Returns:
        float: Correlation coefficient between xs and ys.
    """
    sx = np.std(xs)
    sy = np.std(ys)
    corr = cov(xs, ys) / sx / sy
    return corr


## Chapter 12


def percentile_rows(row_seq, percentiles):
    """Generates a sequence of percentiles from a sequence of rows.

    Args:
        row_seq: sequence of rows to compute percentiles from.
        percentiles: sequence of percentiles to compute.

    Returns:
        ndarray: Array of percentiles for each column in row_seq.
    """
    array = np.asarray(row_seq)
    return np.percentile(array, percentiles, axis=0)


## Chapter 14


def predict(xs, inter, slope):
    """Predicted values of y for given xs.

    Args:
        xs: sequence of x values.
        inter: float intercept of the line.
        slope: float slope of the line.

    Returns:
        ndarray: Predicted y values.
    """
    xs = np.asarray(xs)
    return inter + slope * xs


def fit_line(xs, inter, slope):
    """Fits a line to the given data.

    Args:
        xs: sequence of x values.
        inter: float intercept of the line.
        slope: float slope of the line.

    Returns:
        tuple: (fit_xs, fit_ys) arrays for plotting the fitted line.
    """
    low, high = np.min(xs), np.max(xs)
    fit_xs = np.linspace(low, high)
    fit_ys = predict(fit_xs, inter, slope)
    return fit_xs, fit_ys


def odds(p):
    """Computes odds for a given probability.

    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.

    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define Odds(1)
    to be infinity, so that's what this function does.

    Args:
        p: float probability between 0 and 1.

    Returns:
        float: Odds ratio. Returns infinity if p=1.
    """
    if p == 1:
        return float("inf")
    return p / (1 - p)


def probability(o):
    """Computes the probability corresponding to given odds.

    Example: o=2 means 2:1 odds in favor, or 2/3 probability.

    Args:
        o: float odds, strictly positive.

    Returns:
        float: Probability between 0 and 1.

    Raises:
        ValueError: If o is not positive.
    """
    if o <= 0:
        raise ValueError("Odds must be positive")
    return o / (o + 1)


def probability2(yes, no):
    """Computes the probability corresponding to given odds.

    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.

    Args:
        yes: int or float count of favorable outcomes.
        no: int or float count of unfavorable outcomes.

    Returns:
        float: Probability between 0 and 1.

    Raises:
        ValueError: If yes + no is not positive.
    """
    total = yes + no
    if total <= 0:
        raise ValueError("Total count must be positive")
    return yes / total


def confidence_interval(cdf, percent=90):
    """Compute a confidence interval.

    Args:
        cdf: Cdf object to compute interval from.
        percent: float percent to be included in the interval.

    Returns:
        ndarray: Array containing [lower, upper] bounds of the interval.
    """
    alpha = 1 - percent / 100
    return cdf.inverse([alpha / 2, 1 - alpha / 2])


class Interpolator(object):
    """Represents a mapping between sorted sequences; performs linear interpolation.

    Attributes:
        xs: sorted list of x values.
        ys: sorted list of y values.
    """

    def __init__(self, xs, ys):
        """Initializes the interpolator.

        Args:
            xs: sorted list of x values.
            ys: sorted list of y values.
        """
        self.xs = xs
        self.ys = ys

    def lookup(self, x):
        """Looks up x and returns the corresponding value of y.

        Args:
            x: float value to look up.

        Returns:
            float: Interpolated y value.
        """
        return self._Bisect(x, self.xs, self.ys)

    def reverse(self, y):
        """Looks up y and returns the corresponding value of x.

        Args:
            y: float value to look up.

        Returns:
            float: Interpolated x value.
        """
        return self._Bisect(y, self.ys, self.xs)

    def _Bisect(self, x, xs, ys):
        """Helper function for linear interpolation.

        Args:
            x: float value to interpolate.
            xs: sorted list of x values.
            ys: sorted list of y values.

        Returns:
            float: Interpolated value.
        """
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y


def make_uniform_pmf(low, high, n):
    """Make a uniform Pmf.

    Args:
        low: float lowest value (inclusive).
        high: float highest value (inclusive).
        n: int number of values.

    Returns:
        Pmf: Uniform probability mass function.
    """
    pmf = Pmf()
    for x in np.linspace(low, high, n):
        pmf.set(x, 1)
    pmf.normalize()
    return pmf


def resample(xs, n=None):
    """Draw a sample from xs with the same length as xs.

    Args:
        xs: sequence to resample from.
        n: optional int sample size (default: len(xs)).

    Returns:
        ndarray: Resampled values.
    """
    if n is None:
        n = len(xs)
    return np.random.choice(xs, n, replace=True)


def sample_rows(df, n, replace=False):
    """Choose a sample of rows from a DataFrame.

    Args:
        df: DataFrame to sample from.
        n: int number of rows to sample.
        replace: bool whether to sample with replacement.

    Returns:
        DataFrame: Sampled rows.
    """
    return df.sample(n, replace=replace)


def resample_rows(df):
    """Resamples rows from a DataFrame.

    Args:
        df: DataFrame to resample from.

    Returns:
        DataFrame: Resampled rows with same length as input.
    """
    n = len(df)
    return df.sample(n, replace=True)


def resample_rows_weighted(df, column="finalwgt"):
    """Resamples a DataFrame using probabilities proportional to given column.

    Args:
        df: DataFrame to resample from.
        column: string column name to use as weights.

    Returns:
        DataFrame: Resampled rows with same length as input.
    """
    n = len(df)
    weights = df[column]
    return df.sample(n, weights=weights, replace=True)


def summarize_results(results):
    """Prints the most important parts of linear regression results.

    Args:
        results: RegressionResults object to summarize.
    """
    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print("%s   %0.3g   (%.3g)" % (name, param, pvalue))
    try:
        print("R^2 %.4g" % results.rsquared)
        ys = results.model.endog
        print("Std(ys) %.4g" % ys.std())
        print("Std(res) %.4g" % results.resid.std())
    except AttributeError:
        print("R^2 %.4g" % results.prsquared)


def print_tabular(rows, header):
    """Prints results in LaTeX tabular format.

    Args:
        rows: list of rows to print.
        header: list of strings for column headers.
    """
    s = "\\hline " + " & ".join(header) + " \\\\ \\hline"
    print(s)
    for row in rows:
        s = " & ".join(row) + " \\\\"
        print(s)
    print("\\hline")


class Normal:
    """Represents a Normal distribution."""

    def __init__(self, mu, sigma2, label=""):
        """Initializes a Normal distribution.

        Args:
            mu: float mean of the distribution.
            sigma2: float variance of the distribution.
            label: string label for the distribution.
        """
        self.mu = mu
        self.sigma2 = sigma2
        self.label = label

    def __repr__(self):
        """Returns a string representation."""
        if self.label:
            return "Normal(%g, %g, %s)" % (
                self.mu,
                self.sigma2,
                self.label,
            )
        else:
            return "Normal(%g, %g)" % (self.mu, self.sigma2)

    __str__ = __repr__

    @property
    def sigma(self):
        """Returns the standard deviation.

        Returns:
            float: Standard deviation of the distribution.
        """
        return np.sqrt(self.sigma2)

    def __add__(self, other):
        """Adds a number or other Normal.

        Args:
            other: number or Normal distribution to add.

        Returns:
            Normal: New Normal distribution.
        """
        if isinstance(other, Normal):
            return Normal(
                self.mu + other.mu, self.sigma2 + other.sigma2
            )
        else:
            return Normal(self.mu + other, self.sigma2)

    __radd__ = __add__

    def sample(self, n):
        """Generates a random sample from this distribution.

        Args:
            n: int length of the sample.

        Returns:
            ndarray: Random sample from the distribution.
        """
        sigma = np.sqrt(self.sigma2)
        return np.random.normal(self.mu, sigma, n)

    def __sub__(self, other):
        """Subtracts a number or other Normal.

        Args:
            other: number or Normal distribution to subtract.

        Returns:
            Normal: New Normal distribution.
        """
        if isinstance(other, Normal):
            return Normal(
                self.mu - other.mu, self.sigma2 + other.sigma2
            )
        else:
            return Normal(self.mu - other, self.sigma2)

    __rsub__ = __sub__

    def __mul__(self, factor):
        """Multiplies by a scalar.

        Args:
            factor: float to multiply by.

        Returns:
            Normal: New Normal distribution.
        """
        return Normal(factor * self.mu, factor**2 * self.sigma2)

    __rmul__ = __mul__

    def __div__(self, divisor):
        """Divides by a scalar.

        Args:
            divisor: float to divide by.

        Returns:
            Normal: New Normal distribution.
        """
        return 1 / divisor * self

    __truediv__ = __div__

    def sum(self, n):
        """Return the distribution of the sum of n values.

        Args:
            n: int number of values to sum.

        Returns:
            Normal: Distribution of the sum.
        """
        return Normal(n * self.mu, n * self.sigma2)

    def plot_cdf(self, n_sigmas=4, **options):
        """Plot the CDF of this distribution.

        Args:
            n_sigmas: int how many sigmas to show.
            **options: passed along to plt.plot.
        """
        mu, sigma = self.mu, np.sqrt(self.sigma2)
        low, high = mu - n_sigmas * sigma, mu + 3 * sigma
        xs = np.linspace(low, high, 101)
        ys = scipy.stats.norm.cdf(xs, mu, sigma)
        plt.plot(xs, ys, **options)

    def prob(self, x):
        """Returns the CDF of x.

        Args:
            x: float value to evaluate.

        Returns:
            float: Cumulative probability at x.
        """
        return scipy.stats.norm.cdf(x, self.mu, self.sigma)

    def percentile(self, p):
        """Computes a percentile of a normal distribution.

        Args:
            p: float or sequence of percentiles (0-100).

        Returns:
            float or ndarray: Values at the given percentiles.
        """
        return scipy.stats.norm.ppf(p / 100, self.mu, self.sigma)


def student_cdf(n):
    """Discrete approximation of the CDF of Student's t distribution.

    Args:
        n: int sample size.

    Returns:
        Cdf: Discrete approximation of Student's t CDF.
    """
    ts = np.linspace(-3, 3, 101)
    ps = scipy.stats.t.cdf(ts, df=n - 2)
    rs = ts / np.sqrt(n - 2 + ts**2)
    return Cdf(rs, ps)


def chi_squared_cdf(n):
    """Discrete approximation of the chi-squared CDF with df=n-1.

    Args:
        n: int sample size.

    Returns:
        Cdf: Discrete approximation of chi-squared CDF.
    """
    xs = np.linspace(0, 25, 101)
    ps = scipy.stats.chi2.cdf(xs, df=n - 1)
    return Cdf(ps, xs)


##  Plotting functions


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    Args:
        d: dict to add options to.
        **options: keyword args to add to d.

    Returns:
        dict: Updated dictionary with new key-value pairs.
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like:
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties:
    https://matplotlib.org/api/axes_api.html

    In addition, you can use `legend=False` to suppress the legend.
    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')

    Args:
        **options: keyword arguments for axis properties.
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    Args:
        **options: passed to plt.legend()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    """
    underride(options, loc="best")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)
