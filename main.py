import os
from utils import read_gps_data, write_gps_data
from utils import gps_to_plane, plot_points_sequence, plot_gps_track
from utils import split_track_by_states
from hmm_tour_model import HMM_Tour


data_dir = 'data'
out_dir = 'out'
delta = 20

## Read gps data from files
gpx_files = []
for filename in os.listdir( data_dir ):
    if filename.endswith( '.gpx' ):
        gpx_files.append( os.path.join( data_dir, filename ) )

## Unpack gps data into sequence of 2D points
tracks, points_seq = [], []
for gpx_f in gpx_files:
    for trk in read_gps_data( gpx_f ):
        points = gps_to_plane( trk, delta )
        if ( points.shape[0] >= 2 ):
            tracks.append( trk )
            points_seq.append( points )

## Divide data into train and test data
train_tracks = tracks[1:]
train_points_seq = points_seq[1:]
test_tracks = [ tracks[0] ]
test_points_seq = [ points_seq[0] ]

## Train
hmm = HMM_Tour( delta )
hmm.train( train_points_seq, epochs_count=50, log=True )

## Test
for i in range( len( test_tracks ) ):

    vars_dict, loglik = hmm.infer_vars( test_points_seq[i] )
    states = vars_dict["states"]
    plot_points_sequence( test_points_seq[i], states )
    
    ## Write gpx files
    splits, split_states = split_track_by_states( test_tracks[i], states, delta )
    
    rest_splits = [ splits[i] for i in range( len(splits) ) if split_states[i] == 0 ]
    walk_splits = [ splits[i] for i in range( len(splits) ) if split_states[i] == 1 ]
    car_splits  = [ splits[i] for i in range( len(splits) ) if split_states[i] == 2 ]
    
    write_gps_data( rest_splits, "rest", "{}/track{}-rest.gpx".format( out_dir, i ) )
    write_gps_data( walk_splits, "walk", "{}/track{}-walk.gpx".format( out_dir, i ) )
    write_gps_data( car_splits,  "car",  "{}/track{}-car.gpx".format( out_dir, i ) )

