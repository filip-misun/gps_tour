import numpy as np
from copy import deepcopy
from distribs.uniform_distrib import UniformDistrib, UniformDistribTrainer
from distribs.normal_distrib import NormalDistrib, NormalDistribTrainer, NormalDistribOnlineTrainer
from distribs.joint_indep_distrib import JointIndependentDistrib, JointIndependentDistribTrainer
from distribs.mixture_distrib import MixtureDistrib, MixtureDistribTrainer
from distribs.categorical_distrib import CategoricalDistrib
from distribs.dirac_distrib import DiracDistrib


class HMM_Tour:

    def __init__( self, delta ):

        self.delta = delta

        std_noise = 0.5                     # std of noise in signal (m)

        walk_angle_std = np.pi/8            # std of change in angle in walk
        car_angle_std = np.pi/8             # std of change in angle in car

        avg_walk_vel = 1.4                  # average walking velocity (m/s)
        std_walk_vel = 0.5                  # walk velocity std

        avg_car_vel = 60 * (1000/3600)      # average car velocity (m/s)
        std_car_vel = 30 * (1000/3600)      # car velocity std

        p_walk_straight = 0.9               # probability of straight walk
        p_car_straight = 0.9                # probability of straight car path

        p_stay = 0.99                       # probability of staying in the
                                            # same state
        p_trans = (1-p_stay)/2
        p_rest_init = 1/3
        p_walk_init = 1/3
        p_car_init  = 1/3


        ## Rest distribution
        self.rest_angle_distrib = DiracDistrib( 0 )
        self.rest_dist_distrib = DiracDistrib( 0 )
        self.rest_joint_distrib = DiracDistrib( np.array( [0,0] ) )


        ## Walk distribution
        self.straight_walk_angle_distrib = NormalDistrib( 0, walk_angle_std**2,
                NormalDistribTrainer( mean_fixed=True) )

        self.turn_walk_angle_distrib = UniformDistrib( -np.pi, np.pi, UniformDistribTrainer() )

        self.walk_angle_distrib = MixtureDistrib(
                [self.straight_walk_angle_distrib, self.turn_walk_angle_distrib],
                [p_walk_straight, 1-p_walk_straight],
                MixtureDistribTrainer() )

        self.walk_dist_distrib = NormalDistrib(
                avg_walk_vel*delta, (std_walk_vel*delta)**2,
                NormalDistribTrainer() )

        self.walk_joint_distrib = JointIndependentDistrib(
                [ self.walk_angle_distrib, self.walk_dist_distrib ],
                JointIndependentDistribTrainer() )


        ## Car distribution
        self.straight_car_angle_distrib = NormalDistrib( 0, car_angle_std**2,
                NormalDistribTrainer( mean_fixed=True ) )

        self.turn_car_angle_distrib = UniformDistrib( -np.pi, np.pi, UniformDistribTrainer() )

        self.car_angle_distrib = MixtureDistrib(
                [self.straight_car_angle_distrib, self.turn_car_angle_distrib],
                [p_car_straight, 1-p_car_straight],
                MixtureDistribTrainer() )

        self.car_dist_distrib = NormalDistrib( avg_car_vel*delta, (std_car_vel*delta)**2,
                NormalDistribTrainer() )

        self.car_joint_distrib = JointIndependentDistrib(
                [ self.car_angle_distrib, self.car_dist_distrib ],
                JointIndependentDistribTrainer() )


        ## Noise distrib
        self.noise_dist_distrib = NormalDistrib( 0, std_noise**2, NormalDistribTrainer() )


        # initial state probabilities
        self.init_lik = np.array( [ p_rest_init, p_walk_init, p_car_init ] )

        # emision distributions
        self.emis_distribs = [
                self.rest_joint_distrib,
                self.walk_joint_distrib,
                self.car_joint_distrib ]

        self.angle_distribs = [
                self.rest_angle_distrib,
                self.walk_angle_distrib,
                self.car_angle_distrib ]

        self.dist_distribs = [
                self.rest_dist_distrib,
                self.walk_dist_distrib,
                self.car_dist_distrib ]

        # transition matrix
        self.trans_lik = np.array( [
                [ p_stay,   p_trans,  p_trans ],
                [ p_trans,  p_stay,   p_trans ],
                [ p_trans,  p_trans,  p_stay  ]] )


        self.n_states = 3
        self.init_loglik = np.log( self.init_lik )
        self.trans_loglik = np.log( self.trans_lik )


    def sample( self, n ):
        """
        Sample variables of the model with given length
        """
        init_state_distrib = CategoricalDistrib( self.init_lik )
        trans_state_distribs = [ CategoricalDistrib( self.trans_lik[i] ) for i in range( self.n_states ) ]

        ## Initialize variables of the model
        states   = np.zeros( n, dtype=int )
        alphas   = np.zeros( n )
        dists    = np.zeros( n )
        gammas   = np.zeros( n+1 )
        pos_true = np.zeros( (n+1, 2) )
        pos_obs  = np.zeros( (n+1, 2) )
        errs     = np.zeros( (n, 2) )
        
        pos_true[0] = np.array( [0,0] )
        pos_obs[0]  = np.array( [0,0] )
        gammas[0]    = 0

        # sample first state
        u = init_state_distrib.sample()

        ## Sample variables of the model
        for i in range( 0, n ):
            
            states[i] = u
            alphas[i] = self.angle_distribs[u].sample()
            dists[i] = self.dist_distribs[u].sample()

            gammas[i+1] = gammas[i] + alphas[i]
            direction = np.array( [ np.cos( gammas[i+1] ), np.sin( gammas[i+1] ) ] )
            pos_true[i+1] = pos_true[i] + ( dists[i] * direction )

            errs[i,0] = self.noise_dist_distrib.sample()
            errs[i,1] = self.noise_dist_distrib.sample()

            pos_obs[i+1] = pos_true[i+1] + errs[i]

            u = trans_state_distribs[u].sample()

        # sampled variables dictionary
        vars_dict = {
            "states":   states,
            "alphas":   alphas,
            "dists":    dists,
            "gammas":   gammas,
            "pos_true": pos_true,
            "pos_obs":  pos_obs,
            "errs":     errs }

        return vars_dict


    def infer_vars( self, points ):
        """
        Infer values of hidden variables given a sequence of observed tourist positions.
        """
        points_count = points.shape[0]
        assert ( points_count >= 2 )

        ## Viterbi algorithm
        paths = np.zeros( (points_count-1, self.n_states), dtype=int )
        paths_loglik = [ StatesPathLoglik( self, points[0] ) for i in range( self.n_states ) ]

        for u in range( self.n_states ):
            paths_loglik[u].append_loglik( u, points[1] )

        for i in range( 2, points_count ):
            
            for v in range( self.n_states ):
                logliks = np.zeros( self.n_states )
                paths_loglik_copy = deepcopy( paths_loglik )

                for u in range( self.n_states ):
                    logliks[u] = paths_loglik_copy[u].append_loglik( v, points[i] )

                u = np.argmax( logliks )
                paths[i-1,v] = u
                paths_loglik[v] = paths_loglik_copy[u]

        ## Infer states sequence
        states = np.zeros( points_count-1, dtype=int )
        loglik = np.max( [ pl.loglik for pl in paths_loglik ] )
        states[-1] = np.argmax( [ pl.loglik for pl in paths_loglik ] )

        for i in reversed( range( points_count-2 ) ):
            states[i] = paths[ i+1, states[i+1] ]

        ## Infer alphas, gammas and distances
        dists = np.zeros( points_count-1 )
        alphas = np.zeros( points_count-1 )
        gammas = np.zeros( points_count )
        prev_dx = np.zeros( 2 )

        for i in range( points_count-1 ):
            if ( states[i] == 0 ):
                dist = 0
                d_angle = 0
                prev_dx = np.zeros( 2 )
            else:
                dx = points[i+1] - points[i]
                dist = np.linalg.norm( dx )
                d_angle = self._get_d_angle( prev_dx, dx )
                prev_dx = dx

            dists[i] = dist
            alphas[i] = d_angle
            gammas[i+1] = gammas[i] + d_angle

        ## Infer rest positions
        rest_pos = []
        rest_pos_sum = np.zeros( 2 )
        rest_pos_count = 0
        rest_pos_curr_ind = 0
        rest_pos_ind = np.repeat( -1, points_count )

        for i in range( points_count ):
            if ( ( (i < points_count-1) and (states[i] == 0) )
                    or ( (i>0) and (states[i-1] == 0) ) ):
                rest_pos_sum += points[i]
                rest_pos_count += 1
                rest_pos_ind[i] = rest_pos_curr_ind

            if ( ((i>0) and (i < points_count-1) and (states[i-1] == 0) and (states[i] != 0 )) or 
                    ((i==points_count-1) and (states[i-1] == 0)) ):
                rest_pos.append( rest_pos_sum / rest_pos_count )
                rest_pos_sum = np.zeros( 2 )
                rest_pos_count = 0
                rest_pos_curr_ind += 1

        ## Infer true positions and errors
        errs = [None] * points_count
        pos_true = np.zeros( (points_count,2) )

        for i in range( points_count ):
            if ( rest_pos_ind[i] == -1 ):
                pos_true[i] = points[i]
                errs[i] = None
            else:
                pos_true[i] = rest_pos[ rest_pos_ind[i] ]
                errs[i] = points[i] - pos_true[i]

        # infered variables dictionary
        vars_dict = {
            "states":   states,
            "alphas":   alphas,
            "dists":    dists,
            "gammas":   gammas,
            "pos_true": pos_true,
            "pos_obs":  points,
            "errs":     errs }

        return vars_dict, loglik


    def _train_model_params( self, vars_dicts ):
        """
        Train model parameters when values of variables of the model are known.
        """
        data_count = len( vars_dicts )

        ## Collect alphas, dists, transitions and noise dists
        alphas = [ np.array( [] ) ] * self.n_states
        dists  = [ np.array( [] ) ] * self.n_states
        trans_count = np.zeros( (self.n_states, self.n_states) )
        noise_dists = np.array( [] )

        for i in range( data_count ):

            states_i  = vars_dicts[i]["states"]
            pos_obs_i = vars_dicts[i]["alphas"]
            alphas_i  = vars_dicts[i]["alphas"]
            dists_i   = vars_dicts[i]["dists"]
            errs_i    = vars_dicts[i]["errs"]

            points_count_i = pos_obs_i.shape[0]

            for j in range( points_count_i-2 ):
                u = states_i[j]
                v = states_i[j+1]

                trans_count[u,v] += 1

            for j in range( points_count_i-1 ):
                u = states_i[j]

                alphas[u] = np.append( alphas[u], alphas_i[j] )
                dists[u]  = np.append( dists[u],  dists_i[j] )

            for j in range( points_count_i ):
                e = errs_i[j]
                if ( not e is None ):
                    noise_dists = np.append( noise_dists, np.linalg.norm( e ) )

        ## Train emission distributions
        for u in range( self.n_states ):
            self.dist_distribs[u].train( dists[u] )
        
        for u in range( self.n_states ):
            self.angle_distribs[u].train( alphas[u] )

        ## Train noise distribution
        self.noise_dist_distrib.train( noise_dists )

        ## Train transition distributions
        visit_count = np.sum( trans_count, axis=1 )

        for u in range( self.n_states ):

            if visit_count[u] == 0:
                continue

            for v in range( self.n_states ):

                if trans_count[u,v] == 0:
                    self.trans_loglik[u,v] = -np.inf
                else:
                    self.trans_loglik[u,v] = np.log( trans_count[u,v] ) - np.log( visit_count[u] )


    def train( self, pos_obs, epochs_count=50, log=False ):
        """
        Viterbi training
        """
        data_count = len( pos_obs )
        total_points_count = np.sum( [ points.shape[0] for points in pos_obs ] )

        if ( log ):
            print( "start\n" )
            self._log_model_state()
        
        loglik_eps = 1e-3
        prev_avg_loglik = -np.infty

        for ep in range( epochs_count+1 ):

            vars_dicts = []
            loglik = 0

            for i in range( data_count ):

                vars_dict_i, loglik_i = self.infer_vars( pos_obs[i] )

                vars_dicts.append( vars_dict_i )
                loglik += loglik_i
            
            avg_loglik = loglik / total_points_count
            if ( log ):
                print( "\navg loglik = {:.4f}\n\n\n".format( avg_loglik ) )

            loglik_diff = np.abs( avg_loglik - prev_avg_loglik )
            if ( ep == epochs_count or loglik_diff < loglik_eps ):
                return

            prev_avg_loglik = avg_loglik
            
            self._train_model_params( vars_dicts )

            if ( log ):
                print( "epoch {}/{}\n".format( ep+1, epochs_count ) )
                self._log_model_state()


    def _get_d_angle( self, dx1, dx2 ):
        """
        Calculate angle difference between dx1 and dx2
        """
        dx1_norm = np.linalg.norm( dx1 )
        dx2_norm = np.linalg.norm( dx2 )

        if ( ( dx1_norm < 1e-8 ) or ( dx2_norm < 1e-8 ) ):
             d_angle = 0
        else:
            dx1_normed = dx1 / dx1_norm
            dx2_normed = dx2 / dx2_norm

            orient = np.linalg.det( np.array( [dx1_normed, dx2_normed] ) )
            d_cos_angle = np.clip( np.dot( dx1_normed, dx2_normed ), -1.0, 1.0 )
            d_angle = np.sign( orient ) * np.arccos( d_cos_angle )

        return d_angle


    def _log_model_state( self ):
        """
        Log state of the model
        """
        noise_std = np.sqrt( np.asscalar( self.noise_dist_distrib.covar ) )

        walk_angle_std = np.sqrt( np.asscalar( self.straight_walk_angle_distrib.covar ) )
        walk_mixture = self.walk_angle_distrib.weights
        walk_dist_mean = np.asscalar( self.walk_dist_distrib.mean )
        walk_dist_std = np.sqrt( np.asscalar( self.walk_dist_distrib.covar ) )

        walk_vel_mean = ( walk_dist_mean / self.delta )
        walk_vel_std =  ( walk_dist_std  / self.delta )

        car_angle_std = np.sqrt( np.asscalar(
                self.straight_car_angle_distrib.covar ) )
        car_mixture = self.car_angle_distrib.weights
        car_dist_mean = np.asscalar( self.car_dist_distrib.mean )
        car_dist_std = np.sqrt( np.asscalar( self.car_dist_distrib.covar ) )

        car_vel_mean = ( car_dist_mean / self.delta ) * (3600/1000)
        car_vel_std =  ( car_dist_std  / self.delta ) * (3600/1000)

        trans_lik = np.exp( self.trans_loglik )

        print( "noise std          = {:.3f} m\n".format( noise_std ) )

        print( "walk angle std     = {:.5f}".format(
                walk_angle_std ) )
        print( "walk straight/turn = {:.5f}, {:.5f}".format(
                walk_mixture[0], walk_mixture[1]) )
        print( "walk dist mean     = {:.2f} ({:.2f} m/s)".format(
                walk_dist_mean, walk_vel_mean ) )
        print( "walk dist std      = {:.2f} ({:.2f} m/s)\n".format(
                walk_dist_std, walk_vel_std ) )

        print( "car angle std      = {:.5f}".format( car_angle_std ) )
        print( "car straight/turn  = {:.5f}, {:.5f}".format(
                car_mixture[0], car_mixture[1] ) )
        print( "car dist mean      = {:.2f} ({:.2f} km/s)".format(
                car_dist_mean, car_vel_mean ) )
        print( "car dist std       = {:.2f} ({:.2f} km/s)\n".format(
                car_dist_std, car_vel_std ) )

        np.set_printoptions( precision=4, suppress=True )
        print( "trans lik =\n{}".format( trans_lik ) )
        np.set_printoptions( precision=8, suppress=False )



class StatesPathLoglik:
    
    def __init__( self, hmm_tour, init_pos ):
        
        self.prev_state = -1
        self.prev_pos = np.copy( init_pos )
        self.prev_dx = np.zeros( 2 )

        self.acc_trans_loglik = 0
        self.pre_rest_emis_loglik = 0
        self.rest_emis_loglik = 0

        self.hmm_tour = hmm_tour
        self.init_loglik = hmm_tour.init_loglik
        self.trans_loglik = hmm_tour.trans_loglik
        self.emis_distribs = hmm_tour.emis_distribs

        noise_var = np.asscalar( hmm_tour.noise_dist_distrib.covar )
        self.rest_pos_distrib = NormalDistrib( np.zeros(2), np.diag( [noise_var, noise_var] ),
                NormalDistribOnlineTrainer( covar_fixed=True ) )

        self.loglik = 0


    def append_loglik( self, state, pos ):
        """
        Append given state to the path and return the log likelihood of this path
        given the sequence of observed positions
        """
        dx = pos - self.prev_pos

        ## Transitions loglik

        if ( self.prev_state == -1 ):
            self.acc_trans_loglik = self.init_loglik[ state ]
        else:
            trans_loglik = self.trans_loglik[ self.prev_state, state ]
            self.acc_trans_loglik += trans_loglik


        ## Emission loglik

        if ( state != 0 ):
            ## If appended state isn't rest state

            dist = np.linalg.norm( dx )
            d_angle = self.hmm_tour._get_d_angle( self.prev_dx, dx )

            if ( self.prev_state == 0 ):
                self.pre_rest_emis_loglik += self.rest_emis_loglik
            emis_loglik = self.emis_distribs[ state ].loglik( np.array( [d_angle, dist] ) )

            self.pre_rest_emis_loglik += emis_loglik
            self.rest_emis_loglik = 0

        else:
            ## If appended state is a rest state

            if ( self.prev_state != 0 ):
                ## If previous state wasn't a rest state

                self.rest_pos_distrib.trainer.reset()
                self.rest_pos_distrib.train( self.prev_pos )

            self.rest_emis_loglik = self.rest_pos_distrib.train( pos )

        self.prev_state = state
        self.prev_pos = np.copy( pos )
        self.prev_dx = dx

        emis_loglik = self.pre_rest_emis_loglik + self.rest_emis_loglik
        self.loglik = self.acc_trans_loglik + emis_loglik

        return self.loglik

