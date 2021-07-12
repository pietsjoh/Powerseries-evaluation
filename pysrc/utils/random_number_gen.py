"""This file contains a pseudo-random number generator.
Can create integers, uniform distributed and normal distributed floats.

Attributes
----------
maxInt32: int
    largest signed 32-bit representible integer

eps: float
    precision of floats
"""
import numpy.random as random
import numpy as np
import typing

intOrNone = typing.Union[int, None]
number = typing.Union[float, int, np.number]

maxInt32: int = 2**31 - 1

eps: float = np.finfo(float).eps

class RNGenerator:
    def __init__(self, seed: intOrNone=None, printSeed: bool=True) -> None:
        """A pseudo-random number generator.

        Can be used to generate reliably reproducible random number sets.

        Parameters
        ----------
        seed : int, optional, default = None
            The generators random number seed. Is chosen at random out of the uniform distribution over [0,`maxInt32`) if not specified ('None').
        printSeed : bool, default = True
            Determines whether to print random seed. Useful for debugging purposes.

        Raises
        ------
        AssertionError:
            When seed or printSeed are invalid datatypes, or when seed is out of range [0, maxInt32] 
        """
        seedTmp: int
        if seed is None:
            seedTmp = random.randint(low=0, high=maxInt32)
        else:
            seedTmp = seed
            assert np.issubdtype(type(seedTmp), np.integer)
            assert 0 <= seedTmp <= maxInt32

        assert isinstance(printSeed, bool)
        if printSeed:
            print("random seed:{}".format(seedTmp) )

        self.seed: int = seedTmp
        self.gen = random.default_rng( self.seed )

    def integers(self, low: int=0, high: int=1, size: int=1) -> np.ndarray:
        """Generates an 'np.ndarray' of 'size' integers.
        The integers lie in the range ['low', 'high').

        Parameters
        ----------
        low : int, default = 0
            Lower bound of the random numbers to be generated. Has to be lower than high.
        high : int, default = 1
            Higher bound of the random number to be generated. Has to be higher than low.
        size: int, default = 1
            Number of integers that are generated.

        Returns
        -------
        np.ndarray:
            Array of "size" uniform distributed ints in the range of [low, high).

        Raises
        ------
        AssertionError:
            When low, high or size are invalid datatypes, or when size is below 1, or when low >= high
        """
        assert np.issubdtype(type(low), np.integer)
        assert np.issubdtype(type(high), np.integer)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert low < high

        return self.gen.integers(low=low, high=high, size=size)

    def uniform(self, low: number=0, high: number=1, size: int=1) -> np.ndarray:
        """Generates an 'np.ndarray' of 'size' floats.
        The floats lie in the range ['low', 'high').
        The floats are uniformly distributed.

        Parameters
        ----------
        low : int, default = 0
            Lower bound of the random numbers to be generated. Has to be lower than high.
        high : int, default = 1
            Higher bound of the random number to be generated. Has to be higher than low.
        size: int, default = 1
            Number of floats that are generated.

        Returns
        -------
        np.ndarray:
            Array of "size" uniform distributed floats in the range of [low, high)

        Raises
        ------
        AssertionError:
            When low, high or size are invalid datatypes, or when size is below, or when low >= high
        """
        assert np.issubdtype(type(low), np.integer) or np.issubdtype(type(low), np.floating)
        assert np.issubdtype(type(high), np.integer) or np.issubdtype(type(high), np.floating)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert low < high

        return (high - low)*self.gen.random(size) + low

    def normal(self, mu: number=0, sigma: number=1, size: int=1) -> np.ndarray:
        """Generates an 'np.ndarray' of 'size' floats.
        The floats are distributed according to the normal distribution N(mu, sigma).

        Parameters
        ----------
        mu : float/int, default = 0
            Mean of the underlying normal distribution.
        sigma : float/int, default = 1
            Standard deviation of the underlying normal distribution. Has to be greater than machine precision.
        size: int, default = 1
            Number of floats that are generated.

        Returns
        -------
        np.ndarray:
            Array of "size" normal distributed floats

        Raises
        ------
        AssertionError:
            When mu, sigma or size are invalid datatypes, or when size is below 1 or when sigma <= eps (float precision)
        """
        assert np.issubdtype(type(mu), np.integer) or np.issubdtype(type(mu), np.floating)
        assert np.issubdtype(type(sigma), np.integer) or np.issubdtype(type(sigma), np.floating)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert sigma > eps

        return self.gen.normal(loc=mu, scale=sigma, size=size)