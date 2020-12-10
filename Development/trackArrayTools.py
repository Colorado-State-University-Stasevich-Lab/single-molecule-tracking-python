#### DEPENDENCIES

# To manipulate arrays
import numpy as np 

# To handle track DataFrames
import pandas as pd

# To import images
from skimage import io 
from skimage.io import imread

# To make plots
import matplotlib as mpl 
import matplotlib.pyplot as plt 

# Napari 
from skimage import data
import napari

# To create interactive elements
import ipywidgets as widgets 
from ipywidgets import interact, interactive, fixed, interact_manual, Button, HBox, VBox, Layout, GridspecLayout
from ipywidgets.embed import embed_minimal_html, dependency_state

# Image processing and filters
from skimage.filters import difference_of_gaussians

# Iteration tools such as groupby 
import itertools

# For directories 
import os

from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication


##### CLASSES
class TrackArray:
    """
    A class of track arrays with (1) track array and (2) track data frame
    
    ...
    Attributes
    ----------
    arr : array
        an array containing the 3D tif intensity data
    df : data frame
        a data frame containing the 3D tracks  
            with columns = [TRACK_ID, POSITION_T, POSITION_Z, POSITION Y, POSITION_X]
    crop_pad : int
        the # of pixels padding the tracked pixel in XY 

     """
    def __init__(self, _arr, _df, _crop_pad):
        self.arr = _arr
        self.df = _df
        self.crop_pad = _crop_pad
            
########### Still to be made (ideas welcome!!)

    def save(self):
        return self.df
                        
    def track_classifier(self):
        return self.df
    
    def track_mobility_is(self):
        return self.df
                
############# Made

    def data(self):
        """Returns the track array data corresponding to track array tif file"""
        return self.arr
    
    def tracks(self):
        """Returns the track array data frame corresponding to the track array csv file"""
        return self.df
        
    def crop_dim(self):
        """Returns XY dimensions in pixels of 3D crops in track array"""
        return 2*self.crop_pad + 1
    
    def n_frames(self):
        """Returns the number of time points in track array"""
        return int(self.arr.shape[2]/self.crop_dim())

    def z_slices(self):
        """Returns the Z dimension in pixels of 3D crops in track array"""
        return int(self.arr.shape[0])

    def n_tracks(self):
        """Returns the number of tracks in the track array"""
        return int(self.arr.shape[1]/self.crop_dim())

    def n_channels(self):
        """Returns number of color channels in track array"""
        if len(self.arr.shape) >= 4:
            return int(self.arr.shape[3])
        else:
            return 1

    def track_IDs(self):
        """Returns all the unique track IDs in the track array"""
        # Find the TRACK_ID for every unique track
        return self.df.TRACK_ID.unique()
        
    def int_range(self, sdleft, sdright):   
        """Returns an intensity range list for the track array data for visualization"""
        if self.n_channels() == 1:
            allr = np.ma.masked_equal(self.arr,0).compressed().flatten() # Drop zeros
            r_range = [np.median(allr)-sdleft*np.std(allr), np.median(allr)+sdright*np.std(allr)]
            return r_range
        elif self.n_channels() == 2:
            allr = np.ma.masked_equal(self.arr[:,:,:,0],0).compressed().flatten() # Drop zeros
            allg = np.ma.masked_equal(self.arr[:,:,:,1],0).compressed().flatten() # Drop zeros
            r_range = [np.median(allr)-sdleft*np.std(allr), np.median(allr)+sdright*np.std(allr)]
            g_range = [np.median(allg)-sdleft*np.std(allg), np.median(allg)+sdright*np.std(allg)]
            return [r_range, g_range]
        elif self.n_channels() == 3:
            allr = np.ma.masked_equal(self.arr[:,:,:,0],0).compressed().flatten() # Drop zeros
            allg = np.ma.masked_equal(self.arr[:,:,:,1],0).compressed().flatten() # Drop zeros
            allb = np.ma.masked_equal(self.arr[:,:,:,2],0).compressed().flatten() # Drop zeros            
            r_range = [np.median(allr)-sdleft*np.std(allr), np.median(allr)+sdright*np.std(allr)]
            g_range = [np.median(allg)-sdleft*np.std(allg), np.median(allg)+sdright*np.std(allg)]
            b_range = [np.median(allb)-sdleft*np.std(allb), np.median(allb)+sdright*np.std(allb)]
            return [r_range, g_range, b_range]
        else:
            return "Error: currently only support 1-3 color movies"
                 
    def track_ID_markers(self):
        """Returns a dataframe whose values can be used to label N xT track arrays"""
        n_tracks = self.n_tracks()
        step = self.crop_dim()
        crop_pad = self.crop_pad
        zeros = np.zeros(n_tracks)
        my_track_ids = self.track_IDs()
        return pd.DataFrame(np.array([my_track_ids, zeros, zeros, np.arange(crop_pad, step*n_tracks, step),zeros]).T,
                            columns=['TRACK_ID', 'POSITION_T', 'POSITION_Z', 'POSITION_Y', 'POSITION_X'])
    
    def crops(self):
        """Returns indexable N x T crops from track array."""
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        crops = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim,n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                crops[n,t] = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim] 
        return crops
    
    def best_z(self, crop):    
        """
        Returns crop Z plane with the max average intensity in a central 3x3 square after applying a bandpass filter 
        """
        crop_pad = self.crop_pad
        return np.argmax(np.mean(difference_of_gaussians(crop,1,7)[:,crop_pad-1:crop_pad+2,crop_pad-1:crop_pad+2],axis=(1,2))) 
    
    def z_masks(self, rz, ch):
        """Returns indexable N x T mask for track array with best z +/- zpad = 1, else 0"""
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        crop_pad = self.crop_pad
        best_z_mask = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,ch] # !!! Doesn't work for 1-channel image  
                best_z = self.best_z(cur_crop_3d)
                best_z_mask[n,t] = my_cuboid(crop_pad,crop_pad,best_z,crop_pad,crop_pad,rz,crop_dim,crop_dim,z_slices)
        return best_z_mask.astype('bool')

    def cigar_masks(self, rx, ry, rz):
        """Returns indexable N x T mask for track array with best z +/- zpad = 1, else 0"""
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        crop_pad = self.crop_pad
        cigar_masks = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,0] # !!! Doesn't work for 1-channel image  
                best_z = self.best_z(cur_crop_3d)
                cigar_masks[n,t] = my_cigar(crop_pad,crop_pad,best_z,rx,ry,rz,crop_dim,crop_dim,z_slices) # !!! Fix rx, ry, rz; should be 1,1,1
        return cigar_masks.astype('bool')    

    def capsule_masks(self, rx, ry, rz):
        """Returns indexable N x T mask for track array with best z +/- zpad = 1, else 0"""
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        crop_pad = self.crop_pad
        capsule_masks = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,0] # !!! Doesn't work for 1-channel image  
                best_z = self.best_z(cur_crop_3d)
                capsule_masks[n,t] = my_capsule(crop_pad,crop_pad,best_z,rx,ry,rz,crop_dim,crop_dim,z_slices) # !!! Fix rx, ry, rz; should be 1,1,1
        return capsule_masks.astype('bool')    
    
    def to_crop_array(self, crops):
        """Converts indexable N x T crops/masks to crops/mask array with dimensions (N x crop_pad) x (T x crop_pad) x Z"""
        temp = np.hstack(crops.swapaxes(2,4)).swapaxes(1,3)  # Stacking by N and then by T
        return np.hstack(temp).swapaxes(1,2)

    def to_crop_array_NxZ(self, crops):
        """Converts indexable N x T crops/mask to crops/mask array with dimensions (N x crop_pad) x Z"""
        temp = np.hstack(crops.swapaxes(2,4)).swapaxes(1,3)
        return np.hstack(temp.swapaxes(0,1)).swapaxes(1,2)

    def array_to_crops(self, arr):
        my_axis_t = list(arr.shape).index(self.n_frames()*self.crop_dim())  # to split array by time (column)
        temp=np.array(np.split(arr,self.n_frames(),axis=my_axis_t)) 
        my_axis_n = list(temp.shape).index(self.n_tracks()*self.crop_dim()) # to split array by track number (row)
        return np.array(np.split(temp,self.n_tracks(), axis=my_axis_n))
    
    def to_color_mask(self,masks):
        """Adds n_channels to mask to make a color version"""
        n_channels = self.n_channels()
        return np.moveaxis(np.asarray([masks]*n_channels),0,-1)  # Copy mask for each channel and reorder so channels dimension is last 
    
    def mask_projection(self, mask_array): 
        """Performs max-z projection after applying masks to track_array"""
        n_channels = self.n_channels()
        color_mask = self.to_color_mask(mask_array)
        return np.amax(color_mask*self.arr,0) # multiply arrays and do max_z projection
    
    def napari_viewer(self, arr, spatial_scale, **kwargs): #kwargs are optional arguments, in this case a possible layer or markers
        """View track array w/ napari. Spatial scale must be set. Optional: layer (e.g. mask), markers (e.g. dataframe), and int_range"""
        layer = kwargs.get('layer', np.array([]))
        markers = kwargs.get('markers', pd.DataFrame(np.array([])))
        int_range = kwargs.get('int_range', self.int_range(1,8)) # default range [mean -1 s.d, mean + 8 s.d]  
        my_image = np.moveaxis(arr,-1,0) # !!!only works if n_channels > 1 
        n_channels = self.n_channels()
        ch_colors = ['red','green','blue','gray','magenta']

        viewer = napari.Viewer()
        for i in np.arange(self.n_channels()):
            viewer.add_image(my_image[i], colormap=ch_colors[i],
                         name=ch_colors[i],blending="additive", scale=spatial_scale,
                         contrast_limits=int_range[i])
        if markers.values.any():  # check if markers were specified
            viewer.add_tracks(markers.values, name="TRACK_IDs")
        if layer.any():   # check if a layer was specified
            viewer.add_image(layer, colormap='gray',opacity=0.25,name='layer',blending="additive", scale=spatial_scale)
            
    def find_translating_spots(self, intensities, int_threshhold, run_length):
        """Returns a list of track_ids in which translation above a threshhold intensity and 
        lasting longer than run_length is detected in intensity timeseries"""
        my_id = np.zeros(intensities.shape[0])    # an array to hold the counts
        for i in np.arange(intensities.shape[0]):    # going one track at a time
            s=np.where(intensities[i] > int_threshhold, 1, 0) # 1 if > threshhold, 0 otherwise 
            # Below will create a list of continuous runs of 1s (Int>100) and 0s (Int<100)
            full_listing = [(a, list(b)) for a, b in itertools.groupby(s)]
            # Only take the continuous runs of 1s (Int>100)
            all_runs = [b for a, b in full_listing if a == 1]
            # Cacluate the length of each of these runs
            long_run_lengths = [len(a) for a in all_runs if len(a) >= run_length]  # !!! could improve?
            # Ouput the sum of the lengths of each continous run
            my_id[i] = sum(long_run_lengths)
        # Now count how many times the runs are longer than myRunLength   
        translating_spots0 = np.where(my_id > run_length)[0]
        translating_spots = self.track_IDs()[translating_spots0]
        # Translating spot IDs and the fraction of spots that are translating 
        return translating_spots, translating_spots0

    
    
##### BASIC FUNCTIONS OR METHODS NOT RELYING ON TRACK ARRAY CLASS 

# A cuboid array centered at (cx,cy,cz) with half-lengths (rx,ry,rz) in a volumeXYZ
def my_cuboid(cx,cy,cz,rx,ry,rz,volumeX, volumeY, volumeZ):
    """
    Creates a cube mask centered at (cx,cy,cz) with radii (rx,ry,rz) in volumeX x volumeY x volumeZ 
    """
    x = np.arange(0, volumeX)
    y = np.arange(0, volumeY)
    z = np.arange(0, volumeZ)
    arr = np.zeros((z.size, y.size, x.size))
    stripx = np.heaviside(x[np.newaxis,np.newaxis,:]-(cx-rx),1)-np.heaviside(x[np.newaxis,np.newaxis,:]-(cx+rx+1),1)
    stripy = np.heaviside(y[np.newaxis,:,np.newaxis]-(cy-ry),1)-np.heaviside(y[np.newaxis,:,np.newaxis]-(cy+ry+1),1)
    stripz = np.heaviside(z[:,np.newaxis,np.newaxis]-(cz-rz),1)-np.heaviside(z[:,np.newaxis,np.newaxis]-(cz+rz+1),1)
    mask = stripx*stripy*stripz
    return mask

# An ellipsoid centered at (cx,cy,cz) with semi-axes of rx, ry, and rz in volumeXYZ
# This is basically the 3D version of the 'disk' in disk-donut quantification
def my_cigar(cx,cy,cz,rx,ry,rz,volumeX, volumeY, volumeZ):
    """
    Creates an ellipsoid mask centered at (cx,cy,cz) with radii (rx,ry,rz) in volumeX x volumeY x volumeZ 
    """
    x = np.arange(0, volumeX)
    y = np.arange(0, volumeY)
    z = np.arange(0, volumeZ)
    arr = np.zeros((z.size, y.size, x.size))
    mask = ((1/rx)**2)*(x[np.newaxis,np.newaxis,:]-cx)**2 + ((1/ry)**2)*(y[np.newaxis,:,np.newaxis]-cy)**2 + ((1/rz)**2)*(z[:,np.newaxis,np.newaxis]-cz)**2 < 1
    arr[mask] = 1.
    return arr

# A capsule that surrounds myCigar(cx,cy,cz,rx,ry,rz,volumeX, volumeY, volumeZ)
# This is basically the 3D version of the 'donut' in 'disk-donut' quantification
def my_capsule(cx,cy,cz,rx,ry,rz,volumeX,volumeY,volumeZ):
    """
    Creates a capsule mask centered at (cx,cy,cz) of thickness (rx+1:rx+2,ry+1:ry+1,rz+1:rz+2) in volumeX x volumeY x volumeZ 
    """
    arr1=my_cigar(cx,cy,cz,rx+1,ry+1,rz+1,volumeX, volumeY, volumeZ)
    arr2=my_cigar(cx,cy,cz,rx+2,ry+2,rz+2,volumeX, volumeY, volumeZ)
    return arr2-arr1

def my_cylinder(cx,cy,cz,rx,ry,rz,volumeX,volumeY,volumeZ):
    """
    Creates a cylindrical mask centered at (cx,cy,cz) with radii (rx,ry) and height 2*rz+1 in volumeX x volumeY x volumeZ 
    """
    x = np.arange(0, volumeX)
    y = np.arange(0, volumeY)
    z = np.arange(0, volumeZ)
    arr2D = np.zeros((y.size, x.size))
    mask = ((1/rx)**2)*(x[np.newaxis,:]-cx)**2 + ((1/ry)**2)*(y[:,np.newaxis]-cy)**2  < 1
    arr2D[mask] = 1.
    arr = np.asarray([arr2D]*volumeZ)
    stripz = np.heaviside(z[:,np.newaxis,np.newaxis]-(cz-rz),1)-np.heaviside(z[:,np.newaxis,np.newaxis]-(cz+rz+1),1)
    return arr*stripz

def create_track_array_video(video_directory, video_3D_filename, video_3D_tracks_filename, crop_pad, xy_pixel_size, z_pixel_size):
    """Creates and saves a track array video from a 3D tif video and corresponding track csv file"""
    # Read in 3D video (.tif file) and tracks (.csv file)
    video_3D = imread(video_directory + video_3D_filename)
    tracks = pd.read_csv(video_directory + video_3D_tracks_filename) # Read in tracks are read in as "dataframes (df)"
    # Get dimensions
    n_frames = video_3D.shape[0]
    z_slices = video_3D.shape[1]
    height_y = video_3D.shape[2]
    width_x = video_3D.shape[3]
    if len(video_3D) == 4:     # check if just a single channel video
        n_channels = 1
    else:
        n_channels = video_3D.shape[4]
    # Get unique tracks
    my_track_ids = tracks.TRACK_ID.unique()
    n_tracks = my_track_ids.size
    # Create empty array to hold track array video
    my_crops_all=np.zeros((n_tracks,n_frames,z_slices,2*crop_pad+1,2*crop_pad+1,n_channels))
    # Assign each crop to empty array defined above
    my_i=0
    for my_n in my_track_ids:
        my_track = tracks[(tracks['TRACK_ID'] == my_n) & (tracks['POSITION_X']<width_x-crop_pad-1) 
                & (tracks['POSITION_X']>crop_pad+1) & (tracks['POSITION_Y']<height_y-crop_pad-1) & (tracks['POSITION_Y']>crop_pad+1) ]
        my_times = my_track['POSITION_T'].values.astype(int) 
        my_x = my_track['POSITION_X'].round(0).values.astype(int)
        my_y = my_track['POSITION_Y'].round(0).values.astype(int)
        t_ind = 0
        for t in my_times:
            my_crops_all[my_i,t,:,:,:] = video_3D[t,:,my_y[t_ind]-crop_pad:my_y[t_ind]+crop_pad+1,my_x[t_ind]-crop_pad:my_x[t_ind]+crop_pad+1]
            t_ind = t_ind + 1
        my_i = my_i+1
    # stack all crops into an array shape:
    my_crops_all = np.hstack(my_crops_all.swapaxes(2,4)).swapaxes(1,3) # stack in one dimension
    my_crops_all = np.hstack(my_crops_all).swapaxes(1,2) # stack in the other dimension
    my_crops_all = np.moveaxis(my_crops_all.astype(np.int16),-1,1)   # move channels dim from the end to second for imagej 
    io.imsave(video_directory + video_3D_tracks_filename[:-4] + '_crop_pad_' + str(crop_pad) + '.tif',
            my_crops_all, imagej=True,
            resolution=(1/xy_pixel_size,1/xy_pixel_size),  # store x and y resolution in pixels/nm
            metadata={'spacing':z_pixel_size,'unit':'nm'})  # store z spaxing in nm and set units to nm