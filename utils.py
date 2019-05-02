import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom
import dateutil.parser as dateparser
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


def gps_to_euclid( lon, lat, elev ):
    """
    Longitude, latitude in degrees, elevation in meters.
    """
    earth_rad = 6378137     # equatorial radius of earth (in meters)

    r = elev + earth_rad

    z = np.sin( np.radians( lat ) ) * r
    z_rad = np.sqrt( r**2 - z**2 )

    x = np.cos( np.radians( lon ) ) * z_rad
    y = np.sin( np.radians( lon ) ) * z_rad

    return (x,y,z)


def gps_to_plane( track, delta ):
    """
    Unpack a sequence of gps points into a sequence of points in 2D plane.
    """
    pt_count = len( track )
    if ( pt_count == 0 ):
        return np.array( [] )
    if ( pt_count == 1 ):
        return np.array( [[0,0]] )

    ## Create interpolated function: euclidean position by time
    time_stamps = [ pt["time"] for pt in track ] 
    time_start = np.min( time_stamps )

    pt_time   = np.zeros( pt_count )
    pt_euclid = np.zeros( (pt_count, 3) )

    for i in range( pt_count ):
        pt_time[i] = ( track[i]["time"] - time_start ).total_seconds()
        pt_euclid[i,:] = gps_to_euclid( track[i]["lon"], track[i]["lat"], track[i]["ele"] )
    
    fun_euclid = interpolate.interp1d( pt_time, pt_euclid, axis=0, assume_sorted=False )

    ## Create sequence of euclidean 3D points
    t0 = 0
    t1 = np.max( pt_time )
    t = np.arange( t0, t1, delta )
    t_count  = t.shape[0]
    t_euclid = fun_euclid( t )

    ## Init sequence of euclidean 2D points
    points = np.zeros( (t_count, 2) )
    points[0] = np.array( [0,0] )

    ## Init x_axis, y_axis
    pt0 = t_euclid[0]
    pt0_normed = pt0 / np.linalg.norm( pt0 )
    x_axis = cross( np.array( [0,0,1] ), pt0_normed )

    if ( np.linalg.norm( x_axis ) > 1e-8 ):
        x_axis /= np.linalg.norm( x_axis )
        y_axis = cross( pt0_normed, x_axis )
        y_axis /= np.linalg.norm( y_axis )
    else:
        y_axis = cross( pt0, np.array( [1,0,0] ) )
        y_axis /= np.linalg.norm( y_axis )
        x_axis = cross( y_axis, pt0 )
        x_axis /= np.linalg.norm( x_axis )

    ## Initialize earth and plane transformation matrices
    earth_M = np.array( [ x_axis, y_axis ] )
    plane_M = np.eye( 2 )

    pt_orig = pt0
    plane_pt_orig = np.array( [0,0] )

    ## Unpack sequence of 3D points into a sequence of 2D points
    for i in range( 1, t_count ):
        
        pt = t_euclid[i]
        plane_pt = plane_M @ earth_M @ pt
        points[i] = plane_pt_orig + plane_pt

        # Update earth and plane transformation matrices
        orig_diff = pt - pt_orig
        pt_normed = pt / np.linalg.norm( pt )
        orig_diff_normed = orig_diff / np.linalg.norm( orig_diff )

        x_axis = cross( orig_diff_normed, pt_normed )

        if ( np.linalg.norm( x_axis ) > 1e-5 ):
            
            x_axis /= np.linalg.norm( x_axis )
            y_axis = np.cross( pt_normed, x_axis )
            y_axis /= np.linalg.norm( y_axis )

            earth_M = np.array( [ x_axis, y_axis ] )

            plane_pt_dir = plane_pt / np.linalg.norm( plane_pt )
            plane_M = np.array( [
                [  plane_pt_dir[1], plane_pt_dir[0] ], 
                [ -plane_pt_dir[0], plane_pt_dir[1] ] ] )

            pt_orig = pt
            plane_pt_orig = points[i]

    return points


def read_gps_data( gpx_file ):
    """
    Process gpx file.
    """
    xml_ns = '{http://www.topografix.com/GPX/1/1}'
    tree = ET.parse( gpx_file )
    root = tree.getroot()

    tracks = []

    for trkseg in root.iter( xml_ns + 'trkseg' ):
    
        tracks.append( [] )
    
        for trkpt in trkseg:
            
            pt = {}
            pt["lon"]  = float( trkpt.get('lon') )
            pt["lat"]  = float( trkpt.get('lat') )
            pt["ele"]  = float( trkpt.find( xml_ns + 'ele' ).text )
            pt["time"] = dateparser.parse( trkpt.find( xml_ns + 'time' ).text )
    
            tracks[-1].append( pt )
    
    return tracks


def write_gps_data( tracks, name_prefix, gpx_file ):
    """
    Write gpx file.
    """
    tracks_count = len( tracks )

    root = ET.Element( "gpx" )

    for i in range( tracks_count ):

        trk_elem = ET.SubElement( root, "trk" )
        ET.SubElement( trk_elem, "name" ).text = "{}{}".format( name_prefix, i+1 )

        trkseg_elem = ET.SubElement( trk_elem, "trkseg" )

        for trkpt in tracks[i]:

            trkpt_elem = ET.SubElement( trkseg_elem, "trkpt" )

            lat_str = "{:.6f}".format( trkpt["lat"] )
            lon_str = "{:.6f}".format( trkpt["lon"] )
            ele_str = "{:.3f}".format( trkpt["ele"] )
            time_str = trkpt["time"].strftime( "%Y-%m-%dT%H:%M:%SZ" )

            trkpt_elem.set( "lat", lat_str )
            trkpt_elem.set( "lon", lon_str )

            ET.SubElement( trkpt_elem, "ele"  ).text = ele_str
            ET.SubElement( trkpt_elem, "time" ).text = time_str
    
    tree = ET.ElementTree( root )
    xml_str = ET.tostring( root )

    xml_pretty = xml.dom.minidom.parseString( xml_str )
    xml_pretty_str = xml_pretty.toprettyxml()

    gpx_f = open( gpx_file, "w" )
    gpx_f.write( xml_pretty_str )


def split_track_by_states( track, states, delta ):
    """
    Split gps track by states.
    """
    pt_count = len( track )

    splits = [[]]
    split_state = []

    sort_track = sorted( track, key=lambda x: x["time"] )
    time_start = sort_track[0]["time"]
    current_state = states[0]
    split_state.append( current_state )

    for trkpt in sort_track:
        
        pt_time = ( trkpt["time"] - time_start ).total_seconds()
        state_ind = int( np.round( pt_time / delta ) )
        state_ind = max( state_ind, 0 )
        state_ind = min( state_ind, len(states)-1 )
        
        pt_state = states[ state_ind ]
        if ( pt_state != current_state ):
            splits[-1].append( trkpt )
            splits.append( [] )
            split_state.append( pt_state )

        current_state = pt_state
        splits[-1].append( trkpt )

    return splits, split_state


def plot_gps_track( track, states, delta ):
    """
    Plot a path of GPS data.
    """
    track_len = len( track )
    if ( track_len < 2 ):
        return None

    # Create interpolated functions
    time_stamps = [ pt["time"] for pt in track ] 
    time_start = np.min( time_stamps )

    pt_time   = np.zeros( track_len )
    pt_pos = np.zeros( (track_len, 3) )

    for i in range( track_len ):
        
        pt_time[i] = ( track[i]["time"] - time_start ).total_seconds()
        pt_pos[i] = [ track[i]["lon"], track[i]["lat"], track[i]["ele"] ]
    
    path = interpolate.interp1d( pt_time, pt_pos, axis=0, assume_sorted=False )
    t0, t1 = 0, np.max( pt_time )

    t = np.arange( t0, t1, delta )
    points = path( t )
    points_count = len( points )

    point_states = np.append( states, states[-1] )

    x, y = [], []
    current_state = states[0]
    state_col = ['r','y','g']

    for i in range( points_count ):
        
        x = np.append( x, points[i,0] )
        y = np.append( y, points[i,1] )

        if( (point_states[i] != current_state) or (i == points_count-1) ):

            plt.plot( x, y, state_col[current_state] + '-' )
            plt.plot( x, y, state_col[current_state] + 'o', markersize=2 )
            current_state = point_states[i]
            x = [points[i,0]]
            y = [points[i,1]]

    plt.axis( 'equal' )

    plt.show()


def plot_points_sequence( points, states ):
    """
    Plot a path in 2D plane.
    """
    points_count = points.shape[0]
    point_states = np.append( states, states[-1] )

    x, y = [], []
    current_state = point_states[0]
    state_col = ['r','y','g']

    for i in range( points_count ):
        
        x = np.append( x, points[i,0] )
        y = np.append( y, points[i,1] )

        if( (point_states[i] != current_state) or (i == points_count-1) ):

            plt.plot( x, y, state_col[current_state] + '-' )
            plt.plot( x, y, state_col[current_state] + 'o', markersize=2 )
            current_state = point_states[i]
            x = [points[i,0]]
            y = [points[i,1]]

    plt.axis( 'equal' )
    plt.show()


def cross( u, v ):
    """
    Cross product.
    """
    w = np.array( [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0] ] )

    return w

