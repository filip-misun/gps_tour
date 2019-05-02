import numpy as np


class JointIndependentDistrib:
    
    def __init__( self, distribs, trainer ):
        
        n_distribs = len( distribs )
        assert n_distribs > 0
        
        self.n_distribs = n_distribs
        self.distribs = distribs
        self.trainer = trainer
        trainer.distrib = self

        dims = [ d.dim for d in distribs ]
        self.dim = np.sum( dims )
        self.splits = [ int( np.sum( dims[0:i] ) ) for i in range( self.n_distribs+1 ) ]


    def _split_input( self, x ):
        
        return [ x[ self.splits[i] : self.splits[i+1] ] for i in range( self.n_distribs ) ]


    def _split_inputs( self, xs ):
        
        xs_m = np.atleast_2d( xs )
        return [ xs_m[ :, self.splits[i] : self.splits[i+1] ] for i in range( self.n_distribs ) ]


    def lik( self, x ):
        
        x_split = self._split_input( x )
        x_lik = [ self.distribs[i].lik( x_split[i] ) for i in range( self.n_distribs ) ]

        return np.prod( x_lik )
    

    def loglik( self, x ):
        
        x_split = self._split_input( x )
        x_loglik = [ self.distribs[i].loglik( x_split[i] ) for i in range( self.n_distribs ) ]

        return np.sum( x_loglik )


    def sample( self ):
        
        s = np.array( [] )
        for d in self.distribs:
            s = np.append( s, d.sample() )

        return s
            

    def train( self, xs ):

        return self.trainer.train( xs )


    def mixture_train( self, w, xs ):
        
        return self.trainer.mixture_train( w, xs )


    def __repr__( self ):
        
        return "JointIndependentDistrib(distribs=%r)" % self.distribs


class JointIndependentDistribTrainer:
    
    def __init__( self ):

        self.distrib = None


    def train( self, xs ):
        
        xs_split = self.distrib._split_inputs( xs )

        for i in range( self.distrib.n_distribs ):
            self.distrib.distribs[i].train( xs_split[i] )


    def mixture_train( self, w, xs ):
        
        xs_split = self.distrib._split_inputs( xs )

        for i in range( self.distrib.n_distribs ):
            self.distrib.distribs[i].mixture_train( xs_split[i] )

