"""This file contains a pseudo-random number generator.
Can create integers, uniform distributed and normal distributed floats.
"""
import numpy.random as random
import numpy as np

maxInt32 = 2**31 - 1
"""largest signed 32-bit representible integer
"""

eps = np.finfo(float).eps
"""precision of float
"""



class RandomState:
    def __init__(self, seed=None, printSeed=True):
        """A pseudo-random number generator.

        Can be used to generate reliably reproducible random number sets.

        Parameters
        ----------
        seed : int, optional, default = None
            The generators random number seed. Is chosen at random out of the uniform distribution over [0,`maxInt32`) if not specified ('None').
        printSeed : bool, default = True
            Determines whether to print random seed. Useful for debugging purposes.
        """
        if seed == None:
            seed = random.randint(low=0, high=maxInt32)
        else:
            assert np.issubdtype(type(seed), np.integer)
            assert 0 <= seed <= maxInt32

        assert isinstance(printSeed, bool)
        if printSeed:
            print("random seed:{}".format(seed) )

        self.seed   = seed
        self.gen = random.default_rng( seed )

    def integers(self, low=0, high=1, size=1):
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
        """
        assert np.issubdtype(type(low), np.integer)
        assert np.issubdtype(type(high), np.integer)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert low < high

        return self.gen.integers(low=low, high=high, size=size)

    def uniform(self, low=0, high=1, size=1):
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
        """
        assert np.issubdtype(type(low), np.integer) or np.issubdtype(type(low), np.floating)
        assert np.issubdtype(type(high), np.integer) or np.issubdtype(type(high), np.floating)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert low < high

        return (high - low)*self.gen.random(size) + low

    def normal(self, mu=0, sigma=1, size=1):
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
        """
        assert np.issubdtype(type(mu), np.integer) or np.issubdtype(type(mu), np.floating)
        assert np.issubdtype(type(sigma), np.integer) or np.issubdtype(type(sigma), np.floating)
        assert np.issubdtype(type(size), np.integer)
        assert size >= 1
        assert sigma > eps

        return self.gen.normal(loc=mu, scale=sigma, size=size)

def generator(seed=None):
    """Returns a 'RandomState' random number generator, printing it's random seed.
    """
    return RandomState(seed, True)