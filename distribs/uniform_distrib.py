import numpy as np


class UniformDistrib:

    def __init__( self, low, high, trainer=None ):
        
        if np.isscalar( low ):
            assert np.isscalar( high )
            self.dim = 1
        else:
            assert low.ndim == 1
            assert low.shape == high.shape
            self.dim = len( low )

        assert np.all( low < high )
        
        self.low = low
        self.high = high
        self.trainer = trainer
        self.trainer.distrib = self

        vol = np.prod( high - low )
        self._lik = 1 / vol
        self._loglik = -np.log( vol )


    def _is_in_domain( self, x ):
        
        return np.all( self.low <= x ) and np.all( x <= self.high )


    def lik( self, x ):
        
        if self._is_in_domain( x ):
            return self._lik
        else:
            return 0


    def loglik( self, x ):
        
        if self._is_in_domain( x ):
            return self._loglik
        else:
            return -np.infty


    def train( self, xs ):

        return self.trainer.train( xs )


    def mixture_train( self, w, xs ):
        
        return self.trainer.mixture_train( w, xs )


    def sample( self ):
        
        return np.random.uniform( low=self.low, high=self.high )


    def __repr__( self ):
        
        return "UniformDistrib(lows={}, highs={})".format(
                self.low, self.high)


class UniformDistribTrainer:
    
    def __init__( self ):

        self.distrib = None

    def train( self, xs ):
        pass

    def mixture_train( self, w, xs ):
        pass


class UniformDistribOnlineTrainer:
    
    def __init__( self ):
        
        self.distrib = None
        self.N = 0

    def reset( self ):

        self.N = 0

    def train( self, xs ):
        
        xs_count = len( xs )
        self.N += xs_count
        loglik = self.N * self.distrib._loglik

        return loglik

