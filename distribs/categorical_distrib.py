import numpy as np


class CategoricalDistrib:
    
    def __init__( self, w ):

        assert w.ndim == 1
        self.w = w / np.sum( w )
        self.log_w = np.log( w )
        self.cum_w = np.cumsum( w )
    

    def lik( self, x ):

        return self.w[x]


    def loglik( self, x ):
        
        return self.log_w[x]


    def sample( self ):
        
        toss = np.random.uniform( 0.0, 1.0, 1 )
        c, *_ = np.nonzero( self.cum_w > toss )
        
        return c[0]

    
    def __repr__( self ):
        
        return "CategoricalDistrib(weights={})".format( self.w )

