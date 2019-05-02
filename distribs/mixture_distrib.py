import numpy as np


class MixtureDistrib:

    def __init__( self, distribs, weights, trainer=None ):
        
        n_distribs = len( distribs )
        dims = np.array( [ d.dim for d in distribs ] )
        assert n_distribs > 0
        assert np.all( dims == dims[0] )
        assert np.sum( weights ) > 0

        self.dim = dims[0]
        self.n_distribs = n_distribs
        self.distribs = distribs
        self.weights = weights / np.sum( weights )
        self.trainer = trainer
        self.trainer.distrib = self


    def lik( self, x ):
        
        liks = np.array( [ d.lik( x ) for d in self.distribs ] ) 
        return np.dot( liks, self.weights )
    

    def loglik( self, x ):
        
        lik = self.lik( x )

        if ( lik == 0 ):
            return -np.inf
        else:
            return np.log( lik )
    
    
    def sample( self ):
        
        di = np.argmax( np.random.multinomial( 1, self.weights ) )
        return self.distribs[di].sample()
    

    def train( self, xs ):

        return self.trainer.train( xs )


    def mixture_train( self, w, xs ):
        
        return self.trainer.mixture_train( w, xs )
        

    def __repr__( self ):
        
        return "MixtureDistrib(weights=%r, distribs=%r)" % ( self.weights, self.distribs )


class MixtureDistribTrainer:
    
    def __init__( self ):
        
        self.distrib = None


    def train( self, xs ):
        """
        EM algorithm
        """
        xs_count = len( xs )
        assert xs_count > 0

        e_prev = -np.inf
        n_epochs = 100              #TODO: nastavit konst
        for ep in range( n_epochs ):
            
            # calculate probabilities of inputs in distributions
            liks = np.array( [ [ d.lik( x ) for d in self.distrib.distribs ] for x in xs ] )
            w_liks = self.distrib.weights * liks

            # calculate responsibilities of distributions for inputs
            resp = np.zeros( (xs_count, self.distrib.n_distribs) )
            w_liks_sum = np.sum( w_liks, axis=1 )

            for i in range( xs_count ):

                if w_liks_sum[i] == 0:
                    resp[i] = np.ones( self.distrib.n_distribs ) / self.distrib.n_distribs
                else:
                    resp[i] = w_liks[i] / w_liks_sum[i]

            # train distributions
            for i in range( self.distrib.n_distribs ):
                self.distrib.distribs[i].mixture_train( resp[:,i], xs )

            # adjust weights
            self.distrib.weights = np.sum( resp, axis=0 ) / xs_count

            e = np.mean( [ self.distrib.loglik(x) for x in xs ] )
            if ( np.abs( e - e_prev ) < 0.001 ):        #TODO: nastavit konst
                break

            e_prev = e

