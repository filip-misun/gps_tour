import numpy as np


class DiracDistrib:
    
    def __init__( self, v ):
        
        self.v = v


    def lik( self, x ):
        
        if ( x == self.v ):
            return 1.0
        else:
            return 0.0


    def loglik( self, x ):
        
        if ( x == self.v ):
            return 0.0
        else:
            return -np.inf
    
    
    def sample( self ):
        
        return np.copy( self.v )


    def train( self, xs ):

        pass

