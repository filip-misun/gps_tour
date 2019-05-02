import numpy as np


class NormalDistrib:
    
    def __init__( self, mean, covar, trainer=None ):
        
        m = np.atleast_1d( mean )
        cov = np.atleast_2d( covar )
        dim = len( m )

        assert dim >= 1
        assert m.ndim == 1
        assert cov.shape == ( dim, dim )
        assert np.linalg.matrix_rank( cov ) == dim
        assert np.allclose( cov, cov.T, 1e-8 )

        self.dim = dim
        self.mean = m
        self.covar = cov
        self.prec = np.linalg.inv( cov )

        if trainer:
            self.trainer = trainer
            self.trainer.distrib = self


    def lik( self, x ):
        
        xn = np.atleast_1d( x )
        z = ( 1 / np.sqrt( 2*np.pi ) )**self.dim * np.sqrt( np.linalg.det( self.prec ) )
        p = np.exp( -0.5 * ( ( xn - self.mean ) @ self.prec @ ( xn - self.mean ) ) )

        return z*p


    def loglik( self, x ):
        
        xn = np.atleast_1d( x )
        z = -0.5*self.dim*np.log( 2*np.pi ) + 0.5*np.log( np.linalg.det( self.prec ) )
        p = -0.5*( ( xn - self.mean ) @ self.prec @ ( xn - self.mean ) )

        return z+p
    
    
    def sample( self ):
        
        s = np.random.multivariate_normal( self.mean, self.covar )
        if ( self.dim == 1 ):
            return np.asscalar( s )
        else:
            return s


    def train( self, xs ):

        return self.trainer.train( xs )


    def mixture_train( self, w, xs ):
        
        return self.trainer.mixture_train( w, xs )


    def __repr__( self ):
        
        covar_str = str( self.covar).replace("\n","")
        return "NormalDistrib(mean={}, covar={})".format( self.mean, covar_str )



class NormalDistribTrainer:
    
    def __init__( self, mean_fixed=False, covar_fixed=False ):

        self.mean_fixed = mean_fixed
        self.covar_fixed = covar_fixed
        self.distrib = None
        self._perturb = 1e-6


    def train( self, xs ):
        
        xsn = np.reshape( xs, (-1,self.distrib.dim) )

        if ( not self.mean_fixed ):
            self._train_mean( xsn )

        if ( not self.covar_fixed ):
            self._train_covar( xsn )


    def _train_mean( self, xs ):
        
        if ( len( xs ) == 0 ):
            return False

        self.distrib.mean = np.mean( xs, axis=0 )


    def _train_covar( self, xs ):
        
        xs_count = len( xs )
        if ( xs_count == 0 ):
            return False

        xs_res = xs - self.distrib.mean
        cov = ( xs_res.T @ xs_res ) / xs_count

        if ( np.linalg.matrix_rank( cov ) < self.distrib.dim ):
            cov += self._perturb * np.eye( self.distrib.dim )

        self.distrib.covar = cov
        self.distrib.prec = np.linalg.inv( cov )


    def mixture_train( self, w, xs ):
        
        xsn = np.reshape( xs, (-1,self.distrib.dim) )

        if ( not self.mean_fixed ):
            self._mixture_train_mean( w, xsn )

        if ( not self.covar_fixed ):
            self._mixture_train_covar( w, xsn )


    def _mixture_train_mean( self, w, xs ):
        
        xs_count = len( xs )
        if ( xs_count == 0 ):
            return False

        w_xs = xs * w[:,None]
        self.distrib.mean = np.sum( w_xs, axis=0 ) / np.sum( w )


    def _mixture_train_covar( self, w, xs ):
        
        xs_count = len( xs )
        if ( xs_count == 0 ):
            return False

        xs_res = xs - self.distrib.mean
        cov = np.sum( [ w[i] * np.outer( xs_res[i], xs_res[i] ) for i in
                       range( xs_count) ], axis=0 ) / np.sum( w )

        if ( np.linalg.matrix_rank( cov ) < self.distrib.dim ):
            cov += self._perturb * np.eye( self.distrib.dim )

        self.distrib.covar = cov
        self.distrib.prec = np.linalg.inv( cov )


class NormalDistribOnlineTrainer:
    
    def __init__( self, mean_fixed=False, covar_fixed=False ):

        self.mean_fixed = mean_fixed
        self.covar_fixed = covar_fixed
        self.distrib = None

        self.N = 0
        self.x_sum = 0
        self.x_mah_sum = 0


    def reset( self ):
        
        self.N = 0
        self.x_sum = 0
        self.x_mah_sum = 0


    def train( self, xs ):
        
        xsn = np.reshape( xs, (-1,self.distrib.dim) )

        if ( self.covar_fixed ):
            return self._train_mean_fixed( xsn )


    def _train_mean_fixed( self, xs ):
        
        xs_count = len( xs )
        assert xs_count > 0

        N = self.N + xs_count
        x_sum = self.x_sum + np.sum( xs, axis=0 )
        x_mean = x_sum / N
        x_mah_sum = self.x_mah_sum + np.sum( [ xs[i] @ self.distrib.prec @ xs[i] for i in range(xs_count) ] )

        z = N * ( -0.5*self.distrib.dim*np.log( 2*np.pi ) + 0.5*np.log( np.linalg.det( self.distrib.prec ) ) )
        s = -0.5*( x_mah_sum - N*(x_mean @ self.distrib.prec @ x_mean) )
        loglik = z + s

        self.N = N
        self.x_sum = x_sum
        self.distrib.mean = x_mean
        self.x_mah_sum = x_mah_sum

        return loglik

