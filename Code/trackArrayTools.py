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
        
    def int_range(self, arr, sdleft, sdright):   
        """Returns an intensity range list for the inputted track array arr for visualization. 
        Intensity range is median - sdleft*standard deviation :  median + sdright*standard deviation
        """
        arr_c = np.moveaxis(arr,-1,0) # put channels as first dimension
        n_channels = len(arr_c) # number of color channels
        int_range_out = np.zeros([n_channels,2])
        for ch in np.arange(n_channels):
            intensities = np.ma.masked_equal(arr_c[ch],0).compressed().flatten() # Drop zeros
            int_range_out[ch] = [np.median(intensities)-sdleft*np.std(intensities), 
                         np.median(intensities)+sdright*np.std(intensities)]
        return int_range_out
               
#Correct this so can work with 2D track arrays...invert channels and make list to go through channels so don't have to write again and again..
    def int_range_old(self,sdleft, sdright):   
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
        """Returns a dataframe whose values can be used to label N x T track arrays in Napari"""
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
    
    def crops_to_array_old2(self, crops):
        """Converts indexable N x T crops/masks to crops/mask array with dimensions (N x crop_pad) x (T x crop_pad) x Z"""
        n_dim = len(crops.shape) # Figure if 6D crop array (N,T,Z,Y,X,C) or 5D projection (N,T,Y,X,C)
        if n_dim == 6:
            temp = crops.swapaxes(2,4)       #2 N    T      X       Y    Z   C
            temp0 = np.hstack(temp)          #3 T   (NxX)   Y       Z        C   
            temp1 = temp0.swapaxes(1,3)      #4 T    Z      Y     (NxX)      C  
            output = np.hstack(temp1)        #5 Z   (TxY)  (NxX)             C
        if n_dim == 5:
            output = np.hstack(np.hstack(crops))  
        return output
    
    def crops_to_array_old(self, crops):
        """Converts indexable N x T crops/masks to crops/mask array with dimensions (N x crop_pad) x (T x crop_pad) x Z"""
        n_dim = len(crops.shape) # Figure if 6D crop array (N,T,Z,Y,X,C) or 5D projection (N,T,Y,X,C)
        if n_dim == 6:
            temp = np.hstack(crops.swapaxes(2,4)).swapaxes(1,3)  # Stacking by N and then by T
            output = np.hstack(temp).swapaxes(1,2)       
        if n_dim == 5:
            output = np.hstack(np.hstack(crops.swapaxes(2,3)))  # swap axis so X and Y are not transposed
        return output

    def crops_to_array_NxZ(self, crops): 
        """Converts indexable N x T crops/mask to crops/mask array with dimensions (N x crop_pad) x Z"""
        temp = np.hstack(crops.swapaxes(2,4)).swapaxes(1,3)  # moves Z before stacking
        return np.hstack(temp.swapaxes(0,1)).swapaxes(1,2)  

    def crops_to_array(self,crops):
        """Returns indexable N x T crops from track array."""
        n_tracks = crops.shape[0]
        n_frames = crops.shape[1]
        n_channels = crops.shape[-1]
        crop_dim = crops.shape[-2]
        array_width = n_frames * crop_dim
        array_height = n_tracks * crop_dim
        n_dim = len(crops.shape)
        if n_dim == 6:
            z_slices = crops.shape[2]
            output_arr = np.zeros((z_slices, array_height,array_width,n_channels))
        if n_dim == 5:
            output_arr = np.zeros((array_height,array_width,n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                if n_dim == 6:
                    output_arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim] = crops[n,t]
                if n_dim == 5:
                    output_arr[n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim] = crops[n,t]
        return output_arr
    
    def array_to_crops(self,arr):
        """Returns indexable N x T crops from track array."""
        crop_dim = self.crop_dim()
        n_channels = arr.shape[-1]
        array_width = arr.shape[-2]
        array_height = arr.shape[-3]
        n_tracks = int(array_height/crop_dim)
        n_frames = int(array_width/crop_dim)
        n_dim = len(arr.shape)
        if n_dim == 4:
            z_slices = arr.shape[-4]
            output_crops = np.zeros((n_tracks, n_frames, z_slices, crop_dim, crop_dim ,n_channels))
        if n_dim == 3:
            output_crops = np.zeros((n_tracks, n_frames, crop_dim, crop_dim, n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                if n_dim == 4:
                    output_crops[n,t] = arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim]
                if n_dim == 3:
                    output_crops[n,t] = arr[n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim]
        return output_crops    
    
    def array_to_crops_old(self, arr): 
        """
        Converts a crop array to indexable N x T crops. 
        """
        my_axis_t = list(arr.shape).index(self.n_frames()*self.crop_dim())  # to split array by time (column)
        temp=np.array(np.split(arr,self.n_frames(),axis=my_axis_t)) 
        my_axis_n = list(temp.shape).index(self.n_tracks()*self.crop_dim()) # to split array by track number (row)
        return np.array(np.split(temp,self.n_tracks(), axis=my_axis_n))
    
    def best_z(self, crop, **kwargs):    
        """
        Returns crop Z plane with the max average intensity in a central 3x3 square after applying a bandpass filter. bandpass_cutoffs = [min, max] is an optional argument for the bandpass filter (default = [1,7]).
        """
        [min,max] = kwargs.get('bandpass_cutoffs',[1,7])
        crop_pad = self.crop_pad
        return np.argmax(np.mean(difference_of_gaussians(crop,min,max)[:,crop_pad-1:crop_pad+2,crop_pad-1:crop_pad+2],axis=(1,2))) 
    
    def background_in_mask(self, crop, mask):    
        """
        Returns mean intensity within mask region of the inputted crop.
        """
        myaxis =tuple(np.arange(len(crop.shape))[0:-1]) # Axis for summing mean (channels not included)
        return np.mean(np.ma.masked_equal(crop*mask,0),axis=myaxis).data # mean, ignoring zeros and not summing over channel
    
    def best_z_mask(self, rz, ref_ch, **kwargs):
        """Using image in ref_ch, returns mask for track array with best z +/- rz = 1, else 0.
            z-offset = [z_offset_ch0, z_offset_ch1, ...] is an optional integer list to correct for offsets in z. 
        """
        z_offset = kwargs.get('z_offset', [0 for i in np.arange(self.n_channels())])
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        best_z = np.zeros(n_channels)
        #Create an empty array to hold the mask that matches dimensions of indexable track array crops
        best_z_mask = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim,n_channels)) 
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                # Get the crop using ref_ch that corresponds to n and t from the indexable crops
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,ref_ch] # !!! Doesn't work for 1-channel image  
                ref_z = self.best_z(cur_crop_3d)  ## Find the best z for that crop and set it to ref_z
                for ch in np.arange(n_channels):
                    best_z[ch] = z_offset[ch] + ref_z  ## Adjust ref_z for the other channels by offset
                    if best_z[ch] >= z_slices-1:  ## substract 1 since counting starts from zero
                        best_z[ch] = z_slices-1  ## ensure offset best_z[ch] does no go beyond # of slices
                    elif best_z[ch] < 0:
                        best_z[ch] = 0   ## ensure offset best_z[ch] is not less than zero
                for ch in np.arange(n_channels):
                    # Centered on best_z, make a cuboid mask with radius r_z (r_z = 1 gives best_z +/- 1)
                    best_z_mask[n,t,:,:,:,ch] = my_cuboid(crop_pad,crop_pad,
                                                    best_z[ch],crop_pad,crop_pad,rz,crop_dim,crop_dim,z_slices)
        return self.crops_to_array(best_z_mask.astype('bool'))
    
    def cigar_mask(self, rx, ry, rz, ref_ch, **kwargs):
        """Returns mask from reference channel ref_ch for track array with cigars of dimension rx, ry, and rz centered on best z = 1, else 0. nz-offset is an optional argument that is an integer list of form (z_offset_ch0, z_offset_ch1, ...) 
        """
        z_offset = kwargs.get('z_offset', [0 for i in np.arange(self.n_channels())])
        z_offset = kwargs.get('z_offset', [0 for i in np.arange(self.n_channels())])
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        best_z = np.zeros(n_channels)
        cigar_masks = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim,n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,ref_ch] # !!! Doesn't work for 1-channel image  
                ref_z = self.best_z(cur_crop_3d)
                for ch in np.arange(n_channels):
                    best_z[ch] = z_offset[ch] + ref_z
                    if best_z[ch] >= z_slices-1:  ## substract 1 since counting starts from zero
                        best_z[ch] = z_slices-1  ## ensure offset best_z[ch] does no go beyond # of slices
                    elif best_z[ch] < 0:
                        best_z[ch] = 0   ## ensure offset best_z[ch] is not less than zero
                for ch in np.arange(n_channels):
                    cigar_masks[n,t,:,:,:,ch] = my_cigar(crop_pad,crop_pad,
                                                    best_z[ch],rx,ry,rz,crop_dim,crop_dim,z_slices)
    # !!! Fix rx, ry, rz; should be 1,1,1
        return self.crops_to_array(cigar_masks.astype('bool'))    
    
    def capsule_mask(self, rx, ry, rz, th, ref_ch, **kwargs):
        """Returns mask from reference channel ref_ch for track array with single-pixel width capsule shell of dimensions rx, ry, and rz, thickness th, and centered on best z +/- rz = 1, else 0. nz-offset is an optional argument that is an integer list of form (z_offset_ch0, z_offset_ch1, ...) 
        """
        z_offset = kwargs.get('z_offset', [0 for i in np.arange(self.n_channels())])
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        best_z = np.zeros(n_channels)
        capsule_masks = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim,n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,ref_ch] # !!! Doesn't work for 1-channel image  
                ref_z = self.best_z(cur_crop_3d)
                for ch in np.arange(n_channels):
                    best_z[ch] = z_offset[ch] + ref_z
                    if best_z[ch] >= z_slices-1:  ## substract 1 since counting starts from zero
                        best_z[ch] = z_slices-1  ## ensure offset best_z[ch] does no go beyond # of slices
                    elif best_z[ch] < 0:
                        best_z[ch] = 0   ## ensure offset best_z[ch] is not less than zero
                for ch in np.arange(n_channels):
                    capsule_masks[n,t,:,:,:,ch] = my_capsule(crop_pad,crop_pad,
                                                    best_z[ch],rx,ry,rz,crop_dim,crop_dim,z_slices,th)
    # !!! Fix rx, ry, rz; should be 1,1,1
        return self.crops_to_array(capsule_masks.astype('bool'))   
    
    def cylinder_mask(self, rx, ry, rz, ref_ch, **kwargs):
        """Returns mask from reference channel ref_ch for track array with single-pixel width capsule shell of dimension rx, ry, and rz centered on best z = 1, else 0. nz-offset is an optional argument that is an integer list of form (z_offset_ch0, z_offset_ch1, ...) 
        """
        z_offset = kwargs.get('z_offset', [0 for i in np.arange(self.n_channels())])
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        best_z = np.zeros(n_channels)
        cylinder_masks = np.zeros((n_tracks,n_frames,z_slices,crop_dim,crop_dim,n_channels))
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3d = self.arr[:,n*crop_dim:n*crop_dim+crop_dim,t*crop_dim:t*crop_dim+crop_dim,ref_ch] # !!! Doesn't work for 1-channel image  
                ref_z = self.best_z(cur_crop_3d)
                for ch in np.arange(n_channels):
                    best_z[ch] = z_offset[ch] + ref_z
                    if best_z[ch] >= z_slices-1:  ## substract 1 since counting starts from zero
                        best_z[ch] = z_slices-1  ## ensure offset best_z[ch] does no go beyond # of slices
                    elif best_z[ch] < 0:
                        best_z[ch] = 0   ## ensure offset best_z[ch] is not less than zero
                for ch in np.arange(n_channels):
                    cylinder_masks[n,t,:,:,:,ch] = my_cylinder(crop_pad,crop_pad,
                                                    best_z[ch],rx,ry,rz,crop_dim,crop_dim,z_slices)
    # !!! Fix rx, ry, rz; should be 1,1,1
        return self.crops_to_array(cylinder_masks.astype('bool'))   
    
    def donut_mask_2D(self,r,th):
        """
        Creates a 2D mask array with donuts of radius r and thickness th in each crop.
        """
        inner_disk = my_cylinder(self.crop_pad,self.crop_pad,0,r,r,0,self.crop_dim(),self.crop_dim(),1)
        outer_disk = my_cylinder(self.crop_pad,self.crop_pad,0,r+th,r+th,0,self.crop_dim(),self.crop_dim(),1)
        return self.to_color_mask(np.tile(outer_disk-inner_disk,[self.n_tracks(),self.n_frames()]))[0]
    
    def disk_mask_2D(self,r):
        """
        Creates a 2D mask array with disks of radius r in each crop.
        """
        inner_disk = my_cylinder(self.crop_pad,self.crop_pad,0,r,r,0,self.crop_dim(),self.crop_dim(),1)
        return self.to_color_mask(np.tile(inner_disk,[self.n_tracks(),self.n_frames()]))[0]
    
    def to_color_mask(self,masks):     
        """Adds n_channels to mask to make a color version"""
        n_channels = self.n_channels()
        return np.moveaxis(np.asarray([masks]*n_channels),0,-1)  # Copy mask for each channel and reorder so channels dimension is last 
    
    def mask_projection_old(self, crop_array, mask_array): 
        """Performs max-z projection after applying mask_array to inputted crop_array"""
        temp = np.amax(mask_array*crop_array,axis=0) # problem: should mask zeros when doing max_z projection over z(axis = 0)
                                                                          # otherwise, can end up w/ zeros rather than negative values 
                                                                          # after background subtracting 
        return temp  # set masked zeros back to zero (these are empty parts of array)

    def mask_projection(self, crop_array, mask_array): 
        """Performs max-z projection after applying mask_array to inputted crop_array"""
        minimum = np.min(crop_array)  # find minimum value...can be negative if bg-subtracted crop
        temp = crop_array - minimum   # subtract minimum value so everything is greater or equal to zero (only zero at minimum)
        temp2 = temp*mask_array # now multiply by mask..giving zeros only in mask, positive values elsewhere
        return np.amax(temp2,axis=0) + minimum  # max projection and then add min again to get back original intensities
    
    def local_background_subtract(self, crop_array, mask_array):  ### !!! would be nice to work on something other than self.arr
        """Returns crops after subtracting the background signal measured in masks)
        """
        crops = self.array_to_crops(crop_array) # convert to indexable format w/ dims (N,T,Z,Y,X,C) 
        masks = self.array_to_crops(mask_array) # convert to indexable format w/ dims (N,T,Z,Y,X,C)
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        n_channels = self.n_channels()
        n_dim = len(crops.shape)
        output = np.zeros(crops.shape)
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3D = crops[n,t] 
                background = self.background_in_mask(crops[n,t],masks[n,t])
                for ch in np.arange(n_channels):
                    if n_dim == 6:
                        output[n,t,:,:,:,ch] = crops[n,t,:,:,:,ch] - background[ch] # !!! Doesn't work for single channel image
                    if n_dim == 5:
                        output[n,t,:,:,ch] = crops[n,t,:,:,ch] - background[ch]
        return self.crops_to_array(output)
   
    def local_background_subtract_old(self, masks):  ### !!! would be nice to work on something other than self.arr
        """Returns crops after subtracting the background signal measured in masks)
        """
        crop_dim = self.crop_dim()
        n_tracks = self.n_tracks()
        n_frames = self.n_frames()
        z_slices = self.z_slices()
        crop_dim = self.crop_dim()
        n_channels = self.n_channels()
        crop_pad = self.crop_pad
        output = self.crops()
        for n in np.arange(n_tracks):
            for t in np.arange(n_frames):
                cur_crop_3D = output[n,t] 
                background = self.background_in_mask(cur_crop_3D,masks[n,t])
                for ch in np.arange(n_channels):
                    output[n,t,:,:,:,ch] = cur_crop_3D[:,:,:,ch] - background[ch] # !!! Doesn't work for single channel image
        return output
    
    def moving_avg(self,crop_array,n):
        """
        Returns a n-frame moving average of the orginal crop_array. Final frames for which the moving average cannot be computed are set to zero.
        """
        crops = self.array_to_crops(crop_array)
        crops_ma = crops*0
        for i in np.arange(crops.shape[1]-n):
            crops_ma[:,i] = np.mean(crops[:,i:i+n],axis=1)
        return self.crops_to_array(crops_ma)
    
    def int_renorm_by_row(self, arr, n_sd, top_int):   
        """
        Returns a crop array in which the intensity in each row is renormlized such that an intensity that is n 
        standard deviations beyond the median is set equal to top_int (keeping zero unchanged). 
        """
        # Number of rows:
        n_rows = self.crops().shape[0]
        crop_dim = self.crop_dim()
        out_arr = np.zeros(arr.shape)
        for i in np.arange(0,n_rows*crop_dim,crop_dim):    
            out_arr[:,i:i+crop_dim] = int_renorm(arr[:,i:i+crop_dim], n_sd, top_int) #renormalize each crop array row
        return out_arr
    
    def int_in_mask(self, arr, mask):
        """
        Returns an array of mean intensities in arr within the mask. mask and arr should be 3D (NTZYXC) crop arrays or 2D (NTYXC) crop array. 
        """
        n_dim = len(arr.shape)
        signal=self.array_to_crops(mask)*self.array_to_crops(arr)
        if n_dim == 4:
            output = np.mean(np.ma.masked_equal(signal,0),axis=(2,3,4))  # Find mean, ignoring zeros
        elif n_dim ==3:
            output = np.mean(np.ma.masked_equal(signal,0.),axis=(2,3))   # Find mean, ignoring zeros
        return output.data

    def measure_intensity_in_mask_df(self, arr0, mask, **kwargs):
        '''
        Returns a dataframe with intensities measured in mask for crop array arr. Optional arguments: (1) renorm_frames = [0,1] 
        (default) is the range of frames to use when renormalizing intensity to; (2) start_frame = 0 (default) is used to 
        measure time such that start_frame corresponds to t=0; (3) dt = 1 (default) is the time between frames in minutes; 
        (4) file_number = 0 (default) corresponds to number of file in a filelist the crop array belongs to; 
        (5) replicate_number = 0 (default) corresponds to the replicate number of the file in the crop array list;
        (6) exp_number = 0 (defaul) corresponds to the type of experiment (eg. control would have different number)
        '''
        # get the optional arguments
        renorm_frames = kwargs.get('renorm_frames', [0,1])
        start_frame =  kwargs.get('start_frame', 0)
        dt = kwargs.get('dt', 1)   
        file_number = kwargs.get('file_number', 0)   
        replicate_number = kwargs.get('replicate_number', 0)   
        exp_number = kwargs.get('exp_number', 0)   

        # measured intensities in mask as a numpy array
        arr = self.int_in_mask(arr0,mask) 

        # convert intensity measurements in numpy array to a dataframe 
        norm = np.mean(np.ma.masked_equal(arr[:,renorm_frames[0]:renorm_frames[1],:],0),axis=(0,1)) # intensity renorm. factors
        arr_df = np.zeros((np.prod(arr.shape),11)) # set up an empty array to hold dataframe columns
        row = 0  # counter for keeping track of rows in dataframe
        crop_id = 0  # counter to keep track of crops in array irrespective of the color channel 
        for n in np.arange(arr.shape[0]):
            for f in np.arange(arr.shape[1]):
                for c in np.arange(arr.shape[2]):             # COLUMNS OF DATAFRAME:
                    arr_df[row,0] = exp_number                # type of experiment (e.g. exp. vs. control)
                    arr_df[row,1] = replicate_number          # replicate number in list
                    arr_df[row,2] = file_number               # file number in list
                    arr_df[row,3] = crop_id                   # id for crop  
                    arr_df[row,4] = n                         # crop row in array 
                    arr_df[row,5] = f                         # frame
                    arr_df[row,6] = c                         # channel
                    arr_df[row,7] = f*dt                      # time (assumed in minutes)
                    arr_df[row,8] = (f-start_frame)*dt        # time after harringtonine
                    arr_df[row,9] = arr[n,f,c]                # disk intensity (assumes donut already subtracted)
                    arr_df[row,10] = arr[n,f,c]/norm[c]        # renormalized intensity to ~1 at beginning
                    row = row + 1
                crop_id = crop_id + 1
        # Create dataframe:
        df=pd.DataFrame(arr_df, columns = ['Expt.', 'Rep.','File #','Crop ID','Crop row','Frame','Channel',
                                              'Original time (min)','Time (min)','Intensity (a.u.)','Renorm. Int. (a.u.)'])
        df_filt = df[df['Intensity (a.u.)']!=0]  # filter out zeros (which should only correspond to empty crops)

        return df_filt
    
    def napari_viewer_old(self, arr, spatial_scale, **kwargs): #kwargs are optional arguments, in this case a possible layer or markers
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
    stripx = np.heaviside(x[np.newaxis,np.newaxis,:]-(cx-rx),1)-np.heaviside(x[np.newaxis,np.newaxis,:]-(cx+rx),0)
    stripy = np.heaviside(y[np.newaxis,:,np.newaxis]-(cy-ry),1)-np.heaviside(y[np.newaxis,:,np.newaxis]-(cy+ry),0)
    stripz = np.heaviside(z[:,np.newaxis,np.newaxis]-(cz-rz),1)-np.heaviside(z[:,np.newaxis,np.newaxis]-(cz+rz),0)
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
    mask = ((1/rx)**2)*(x[np.newaxis,np.newaxis,:]-cx)**2 + ((1/ry)**2)*(y[np.newaxis,:,np.newaxis]-cy)**2 + ((1/rz)**2)*(z[:,np.newaxis,np.newaxis]-cz)**2 <= 1
    arr[mask] = 1.
    return arr

# A capsule that surrounds myCigar(cx,cy,cz,rx,ry,rz,volumeX, volumeY, volumeZ)
# This is basically the 3D version of the 'donut' in 'disk-donut' quantification
def my_capsule(cx,cy,cz,rx,ry,rz,volumeX,volumeY,volumeZ,th):
    """
    Creates a capsule mask centered at (cx,cy,cz) of thickness (rx+1:rx+2,ry+1:ry+1,rz+1:rz+2) in volumeX x volumeY x volumeZ 
    """
    arr1=my_cigar(cx,cy,cz,rx,ry,rz,volumeX, volumeY, volumeZ)
    arr2=my_cigar(cx,cy,cz,rx+th,ry+th,rz+th,volumeX, volumeY, volumeZ)
    return arr2-arr1

def my_cylinder(cx,cy,cz,rx,ry,rz,volumeX,volumeY,volumeZ):
    """
    Creates a cylindrical mask centered at (cx,cy,cz) with radii (rx,ry) and height 2*rz+1 in volumeX x volumeY x volumeZ 
    """
    x = np.arange(0, volumeX)
    y = np.arange(0, volumeY)
    z = np.arange(0, volumeZ)
    arr2D = np.zeros((y.size, x.size))
    mask = ((1/rx)**2)*(x[np.newaxis,:]-cx)**2 + ((1/ry)**2)*(y[:,np.newaxis]-cy)**2  <= 1
    arr2D[mask] = 1.
    arr = np.asarray([arr2D]*volumeZ)
    stripz = np.heaviside(z[:,np.newaxis,np.newaxis]-(cz-rz),1)-np.heaviside(z[:,np.newaxis,np.newaxis]-(cz+rz),0)
    return arr*stripz

def int_renorm(arr, n, top_int):
    """
    Returns a renormalized array in which the intensity bin corresponding to n standard deviations beyond mean 
    is equal to top_int (keeping zero unchanged):
    """
    # Renormalize and plot all track arrays together
    arr_c = np.moveaxis(arr,-1,0) # put channels as first dimension
    arr_renorm = np.zeros(arr_c.shape)
    n_channels = len(arr_c) # number of color channels
    my_mean = [np.mean(np.ma.masked_equal(arr_c[ch],0)) for ch in np.arange(n_channels)]
    my_std = [np.std(np.ma.masked_equal(arr_c[ch],0)) for ch in np.arange(n_channels)]
    # Renormalize so bin corresponding to n standard deviations beyond mean is renormalized 
    #to top_int (keeping zero unchanged):
    for ch in np.arange(n_channels):
        arr_renorm[ch] = (top_int/(my_mean[ch] + n*my_std[ch]))*arr_c[ch]
    return np.moveaxis(arr_renorm,0,-1)

def create_track_array_video_old(output_directory, output_filename, video_3D, tracks, crop_pad, xy_pixel_size, z_pixel_size):
    """Creates and saves a track array video at output_direction/output_filename from a 3D tif video (video_3D) and corresponding track dataframe (tracks). crop_pad is the effective radius of crops in the generated track array. xy_pixel_size and z_pixel_size are included to generate an imagej tif file with metadata containing the resolution of the image. """
    # Get dimensions...usually t, z, y, x, c. However, can be tricky if channels in a weird place. I assume
    # the smallest dimension is channels and remove it. I then assume remaining is t,z,y,x.  
    dims = list(video_3D.shape)
    if len(dims) == 4:     # check if just a single channel video
        n_channels = 1
        n_frames, z_slices, height_y, width_x = dims
    else:
        n_channels = min(dims)
        n_channels_index = dims.index(n_channels)   # find index of n_channels, which is assumed to be smallest dimension 
        dims.remove(n_channels)    
        video_3D = np.moveaxis(video_3D,n_channels_index,-1)  # move channels to last dimension of array (assumed by napari)
        n_frames, z_slices, height_y, width_x = dims
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
        my_x = my_track['POSITION_X'].round(0)
        my_y = my_track['POSITION_Y'].round(0)
        t_ind = 0
        for t in my_times:
            my_crops_all[my_i,t,:,:,:] = video_3D[t,:,my_y[t_ind]-crop_pad:my_y[t_ind]+crop_pad+1,my_x[t_ind]-crop_pad:my_x[t_ind]+crop_pad+1]
            t_ind = t_ind + 1
        my_i = my_i+1
    # stack all crops into an array shape:
    my_crops_all = np.hstack(my_crops_all.swapaxes(2,4)).swapaxes(1,3) # stack in one dimension
    my_crops_all = np.hstack(my_crops_all).swapaxes(1,2) # stack in the other dimension
    my_crops_all = np.moveaxis(my_crops_all.astype(np.int16),-1,1)   # move channels dim from the end to second for imagej 
    io.imsave(output_directory + output_filename,
            my_crops_all, imagej=True,
            resolution=(1/xy_pixel_size,1/xy_pixel_size),  # store x and y resolution in pixels/nm
            metadata={'spacing':z_pixel_size,'unit':'nm'})  # store z spaxing in nm and set units to nm
    
def create_track_array_video(output_directory, output_filename, video_3D, tracks, crop_pad, xy_pixel_size, z_pixel_size,**kwargs):
    """Creates and saves a track array video at output_direction/output_filename from a 3D tif video (video_3D) and corresponding track dataframe (tracks). crop_pad is the effective radius of crops in the generated track array. xy_pixel_size and z_pixel_size are included to generate an imagej tif file with metadata containing the resolution of the image. An optional argument, homography, is a homography matrix that shifts red (channel 0) pixels so they align with other channels. This will correct for shifts in red and green channels."""
    # Get homography matrix; default is identity matrix
    homography = kwargs.get('homography', np.eye(3))
    # Get dimensions...usually t, z, y, x, c. However, can be tricky if channels in a weird place. I assume
    # the smallest dimension is channels and remove it. I then assum remaining is t,z,y,x.  
    dims = list(video_3D.shape)
    if len(dims) == 4:     # check if just a single channel video
        n_channels = 1
        n_frames, z_slices, height_y, width_x = dims
    else:
        n_channels = min(dims)
        n_channels_index = dims.index(n_channels)   # find index of n_channels, which is assumed to be smallest dimension 
        dims.remove(n_channels)    
        video_3D = np.moveaxis(video_3D,n_channels_index,-1)  # move channels to last dimension of array (assumed by napari)
        n_frames, z_slices, height_y, width_x = dims
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

        ## Use homology to correct x's and y's from different channels
        my_x = np.zeros((n_channels, my_track['POSITION_X'].size))
        my_y = np.zeros((n_channels, my_track['POSITION_Y'].size))
        
        for ch in np.arange(n_channels):  
            if ch == 0:  # don't correct channel 0 (red channel)
                my_x[ch] = my_track['POSITION_X'].round(0).values.astype(int)
                my_y[ch] = my_track['POSITION_Y'].round(0).values.astype(int)
            else:   # correct other channels using homography (since green/blue are image on same camera)
                temp = [list(np.dot(homography,np.array([pos[0],pos[1],1]))[0:2]) 
                        for pos in my_track[['POSITION_X','POSITION_Y']].values]
                my_x[ch], my_y[ch] = np.array(temp).T
                my_x[ch] = my_x[ch].round(0).astype(int)
                my_y[ch] = my_y[ch].round(0).astype(int)

        ## Assign crops        
        t_ind = 0
        for t in my_times:
            for ch in np.arange(n_channels):
                my_crops_all[my_i,t,:,:,:,ch] = video_3D[t,:,my_y[ch,t_ind].astype(int)-crop_pad:my_y[ch,t_ind].astype(int)+crop_pad+1,my_x[ch,t_ind].astype(int)-crop_pad:my_x[ch,t_ind].astype(int)+crop_pad+1,ch]
            t_ind = t_ind + 1
        my_i = my_i+1

    # stack all crops into an array shape:
    my_crops_all = np.hstack(my_crops_all.swapaxes(2,4)).swapaxes(1,3) # sta|ck in one dimension
    my_crops_all = np.hstack(my_crops_all).swapaxes(1,2) # stack in the other dimension
    my_crops_all = np.moveaxis(my_crops_all.astype(np.int16),-1,1)   # move channels dim from the end to second for imagej 

    # write out track array file to directory
    io.imsave(output_directory + output_filename,
            my_crops_all, imagej=True,
            resolution=(1/xy_pixel_size,1/xy_pixel_size),  # store x and y resolution in pixels/nm
            metadata={'spacing':z_pixel_size,'unit':'nm'})  # store z spaxing in nm and set units to nm
    
def create_particle_array_video(output_directory, output_filename, video_3D, particles, 
                                crop_pad, xy_pixel_size, z_pixel_size,**kwargs):
    """Creates and saves a particle array video at output_direction/output_filename from a 
    3D tif video (video_3D) and corresponding particle array dataframe (particles). crop_pad is 
    the effective radius of crops in the generated particle array. xy_pixel_size and z_pixel_size 
    are included to generate an imagej tif file with metadata containing the resolution of 
    the image. An optional argument, homography, is a homography matrix that shifts red 
    (channel 0) pixels so they align with other channels. This will correct for shifts in red 
    and green channels."""
    # Get homography matrix; default is identity matrix
    homography = kwargs.get('homography', np.eye(3))
    homographies = [homography]
    # Get dimensions...usually t, z, y, x, c. However, can be tricky if channels in a weird place. I assume
    # the smallest dimension is channels and remove it. I then assum remaining is t,z,y,x.  
    dims = list(video_3D.shape)
    if len(dims) == 4:     # check if just a single channel video
        n_channels = 1
        n_frames, z_slices, height_y, width_x = dims
    else:
        n_channels = min(dims)
        n_channels_index = dims.index(n_channels)   # find index of n_channels, which is assumed to be smallest dimension 
        dims.remove(n_channels)    
        video_3D = np.moveaxis(video_3D,n_channels_index,-1)  # move channels to last dimension of array (assumed by napari)
        n_frames, z_slices, height_y, width_x = dims
        
    # Special for particle arrays:
    my_particle_ids = particles.TRACK_ID.unique()
    particles_time = particles.groupby('POSITION_T')
    my_times = np.array([i for i in particles_time.groups.keys()])
    n_particles_per_frame = np.array([len(particles_time.groups[i]) for i in my_times])
    n_particles_max = np.max(n_particles_per_frame)
    
    # Create empty array to hold track array video
    my_crops_all = np.zeros((n_particles_max,n_frames,z_slices,2*crop_pad+1,2*crop_pad+1,n_channels))
    
# Assign each crop to empty array defined above
    my_t = 0
    for t in my_times:
        # make sure the 3D crop will not extend beyond the boundaries of the original 3D image
        # I add another 2 pixels too, just in case the homography registration doesn't push points in one color off the image.
        my_col = particles[(particles['POSITION_T'] == t) & (particles['POSITION_X']<width_x-crop_pad-5) 
                & (particles['POSITION_X']>crop_pad+5) & (particles['POSITION_Y']<height_y-crop_pad-5) & (particles['POSITION_Y']>crop_pad+5) ]
        my_IDs = my_col['TRACK_ID'].values.astype(int) 
        my_x = np.zeros((n_channels, my_col['POSITION_X'].size))
        my_y = np.zeros((n_channels, my_col['POSITION_Y'].size))
        # use the homography to correct channels 1 and 2 (assumed channel 0 is red channel)        
        for ch in np.arange(n_channels):
            if ch == 0:  # don't correct channel 0
                my_x[ch] = my_col['POSITION_X'].round(0).values.astype(int)
                my_y[ch] = my_col['POSITION_Y'].round(0).values.astype(int)
            else:   # correct other channels using same homography (since green/blue are image on same camera)
                temp = [list(np.dot(homography,np.array([pos[0],pos[1],1]))[0:2]) 
                        for pos in my_col[['POSITION_X','POSITION_Y']].values]
                my_x[ch], my_y[ch] = np.array(temp).T
                my_x[ch] = my_x[ch].round(0).astype(int)
                my_y[ch] = my_y[ch].round(0).astype(int) 
        for i in np.arange(len(my_IDs)):
            for ch in np.arange(n_channels):
                # create all 3D crops in track array using corrected x and y values:
                my_crops_all[i,my_t,:,:,:,ch] = video_3D[my_t,:,
                        my_y[ch,i].astype(int)-crop_pad:my_y[ch,i].astype(int)+crop_pad+1,
                        my_x[ch,i].astype(int)-crop_pad:my_x[ch,i].astype(int)+crop_pad+1,ch]
        my_t = my_t + 1

    # stack all crops into an array shape:
    my_crops_all = np.hstack(my_crops_all.swapaxes(2,4)).swapaxes(1,3) # stack in one dimension
    my_crops_all = np.hstack(my_crops_all).swapaxes(1,2) # stack in the other dimension
    my_crops_all = np.moveaxis(my_crops_all.astype(np.int16),-1,1)   # move channels dim from the end to second for imagej 

    # save to file
    io.imsave(output_directory + output_filename,
            my_crops_all, imagej=True,
            resolution=(1/xy_pixel_size,1/xy_pixel_size),  # store x and y resolution in pixels/nm
            metadata={'spacing':z_pixel_size,'unit':'nm'})  # store z spaxing in nm and set units to nm
    
    
def concat_crop_array_vids(ca_vids):
    """
    Returns a single, large crop array video made up from a vertical stack of the inputted crop array videos 
    ca_vids = [ca_vid1, ca_vid2, ...]. ca_vids should all have the same crop_pad size and number of dimensions 
    (either 3D+color = ZYXC or 2D+color = YXC).
    """
    # Find the number of dimensions of the crop array videos
    n_dims = len(ca_vids[0].shape)

    if n_dims == 4:
        dims=np.array([i.shape for i in ca_vids])
        ca_all = np.zeros((np.max(dims[:,0]),sum(dims[:,1]),np.max(dims[:,2]),dims[0,3]))    
        for i in np.arange(len(dims)):
            ca_all[0:dims[i,0],sum(dims[0:i,1]):sum(dims[0:i+1,1]),0:dims[i,2],:]=ca_vids[i]
        output = ca_all
    elif n_dims == 3:
        dims=np.array([i.shape for i in ca_vids])
        ca_all = np.zeros((sum(dims[:,0]),np.max(dims[:,1]),dims[0,2]))    
        for i in np.arange(len(dims)): 
            ca_all[sum(dims[0:i,0]):sum(dims[0:i+1,0]),0:dims[i,1],:]=ca_vids[i]
        output = np.array([ca_all]) # Makes a z-dimension of size 1
    else:
        'Error: this function only support 3D+color or 2D+color track arrays.'

        # Make giant track array object containing all track arrays and tracks:
    return output

def napari_viewer(arr, spatial_scale, **kwargs): #kwargs are optional arguments, in this case a possible layer or markers
    """View track array w/ napari. Spatial scale must be set. Optional: layer (e.g. mask), markers (e.g. dataframe), and range"""
    layer = kwargs.get('layer', np.array([[]]))
    markers = kwargs.get('markers', pd.DataFrame(np.array([])))
    int_range = kwargs.get('int_range', [[0,np.max(arr)],[0,np.max(arr)],[0,np.max(arr)]]) # default range [median -1 s.d, median + 8 s.d]  
    my_image = np.moveaxis(arr,-1,0) # !!!only works if n_channels > 1 
    n_channels = len(my_image)
    ch_colors = ['red','green','blue','gray','magenta']

    viewer = napari.Viewer()
    for i in np.arange(n_channels):
        viewer.add_image(my_image[i], colormap=ch_colors[i],
                     name=ch_colors[i],blending="additive", scale=spatial_scale,
                     contrast_limits=int_range[i])
    if markers.values.any():  # check if markers were specified
        viewer.add_tracks(markers.values, name="IDs")
    if layer[0].any():   # check if a layer was specified
        for i in np.arange(len(layer)):
            viewer.add_image(layer[i], colormap=ch_colors[i],opacity=0.25,name='layer',blending="additive", scale=spatial_scale) 
            


##### BASIC FUNCTIONS FOR PLOTTING DATA

def my_mean_intensity_plot(int, **kwargs):
    """
    Plot mean track intensity (averaging columns) from track array intensities. Optional arguments: channels = [0,1,..], xlim = [xmin,xmax],     ylim = [ymin, ymax], colors = ['red', 'green', ...], markers = ['o','s','v', ...], labels = ['ch1', 'ch2', ...], renorm = False,
    filename = 'filename.svg', style = 'seaborn-whitegrid', aspect_ratio = default, error = 'sd' (or 'sem'). 
    Notes: (1) If renorm = True, plots are renormalized to one at first timepoint 
    """
    n_channels = int.shape[-1]
    n_frames = int.shape[1]
    int_c = np.moveaxis(int,-1,0)
    
    channels = kwargs.get('channels', [i for i in np.arange(n_channels)]) 
    xlim = kwargs.get('xlim', [0,0])
    ylim = kwargs.get('ylim', [0,0])
    colors = kwargs.get('colors', ['tab:red', 'tab:green', 'tab:blue','tab:orange','tab:purple'])
    markers = kwargs.get('markers', ['o','s','v','^','d'])
    labels = kwargs.get('labels', ['ch1','ch2','ch3','ch4','ch5'])
    renorm = kwargs.get('renorm', [0,0])
    filename = kwargs.get('filename', 'none')
    style = kwargs.get('style', 'seaborn-whitegrid')
    aspect_ratio = kwargs.get('aspect_ratio',0)
    error = kwargs.get('error','sd')
    times = kwargs.get('times', np.array([]))
    axeslabels = kwargs.get('axeslabels',['ch1','ch2','ch3','ch4','ch5'])
    
    plt.style.use(style)
    
    my_int = np.zeros([n_channels, n_frames])    
    my_sd = np.zeros([n_channels, n_frames]) 
    my_sem = np.zeros([n_channels, n_frames]) 
    my_int_renorm = np.zeros([n_channels, n_frames])  
    my_sd_renorm = np.zeros([n_channels, n_frames])
    my_sem_renorm = np.zeros([n_channels, n_frames])
    
    fig, ax = plt.subplots()

    if times.any():
        t = times
    else:
        t = np.arange(n_frames)

    ch_index=0

    ax.set_xlabel(axeslabels[0])
    ax.set_ylabel(axeslabels[1])
    ax.set_title('Mean Intensity vs. time')

    for ch in channels:
        my_int[ch] = np.mean(int_c[ch],axis=0)
        my_sd[ch] = np.std(int_c[ch],axis=0)
        my_sem[ch] = np.std(int_c[ch],axis=0)/np.sqrt(int.shape[0])
        if error == 'sem':
            yerror = my_sem[ch]
            yerror_renorm = my_sem_renorm[ch]
        else:
            yerror = my_sd[ch]
            yerror_renorm = my_sd_renorm[ch]
        if renorm == [0,0]:
            ax.plot(t, my_int[ch], color=colors[ch_index], marker=markers[ch_index],label=labels[ch_index])
            ax.errorbar(t,my_int[ch],yerr=yerror,color=colors[ch_index],capsize=3)
        elif renorm != [0,0]: 
            my_int_renorm[ch] = my_int[ch]/np.mean(my_int[ch,renorm[0]:renorm[1]])
            my_sd_renorm[ch] = my_sd[ch]/np.mean(my_int[ch,renorm[0]:renorm[1]])
            my_sem_renorm[ch] = my_sem[ch]/np.mean(my_int[ch,renorm[0]:renorm[1]])
            ax.plot(t, my_int_renorm[ch], color=colors[ch_index], marker=markers[ch_index],label=labels[ch_index])
            ax.errorbar(t,my_int_renorm[ch],yerr=yerror_renorm[ch],color=colors[ch_index],capsize=3)            
        ch_index = ch_index + 1

    if xlim != [0,0]:
        plt.xlim(xlim[0], xlim[1])
    if ylim != [0,0]:
        plt.ylim(ylim[0], ylim[1])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if aspect_ratio != 0:
        ax.set_aspect(aspect_ratio)        
        
    if filename != 'none':
        plt.savefig(filename, format = 'svg', dpi=300)
        if len(channels) == 1:
            str1 = 'Mean Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str2 = 'SD of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str3 = 'SEM of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            if renorm != [0,0]:
                dfout=pd.DataFrame({str1:my_int[channels[0]],str2:my_sd[channels[0]],str3:my_sem[channels[0]]})
            elif renorm == [0,0]:
                dfout=pd.DataFrame({str1:my_int_renorm[channels[0]],
                                    str2:my_sd_renorm[channels[0]],str3:my_sem_renorm[channels[0]]})
        if len(channels) == 2:
            str1 = 'Mean Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str2 = 'SD of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str3 = 'SEM of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str4 = 'Mean Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            str5 = 'SD of Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            str6 = 'SEM of Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            if renorm != [0,0]:
                dfout=pd.DataFrame({str1:my_int[channels[0]],str2:my_sd[channels[0]],str3:my_sem[channels[0]],
                                str4:my_int[channels[1]],str5:my_sd[channels[1]],str6:my_sem[channels[1]]})
            elif renorm == [0,0]:
                dfout=pd.DataFrame({str1:my_int_renorm[channels[0]],str2:my_sd_renorm[channels[0]],str3:my_sem_renorm[channels[0]],
                                str4:my_int_renorm[channels[1]],str5:my_sd_renorm[channels[1]],str6:my_sem_renorm[channels[1]]})                
        if len(channels) == 3:
            str1 = 'Mean Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str2 = 'SD of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str3 = 'SEM of Int. Ch ' + str(channels[0]) + ' (' + labels[0] + '; a.u.)'
            str4 = 'Mean Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            str5 = 'SD of Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            str6 = 'SEM of Int. Ch ' + str(channels[1]) + ' (' + labels[1] + '; a.u.)'
            str7 = 'Mean Int. Ch ' + str(channels[2]) + ' (' + labels[2] + '; a.u.)'
            str8 = 'SD of Int. Ch ' + str(channels[2]) + ' (' + labels[2] + '; a.u.)'
            str9 = 'SEM of Int. Ch ' + str(channels[2]) + ' (' + labels[2] + '; a.u.)'
            if renorm != [0,0]:
                dfout=pd.DataFrame({str1:my_int[channels[0]],str2:my_sd[channels[0]],str3:my_sem[channels[0]],
                                str4:my_int[channels[1]],str5:my_sd[channels[1]],str6:my_sem[channels[1]],
                                str7:my_int[channels[2]],str8:my_sd[channels[2]],str9:my_sem[channels[2]]})
            elif renorm == [0,0]:
                dfout=pd.DataFrame({str1:my_int_renorm[channels[0]],str2:my_sd_renorm[channels[0]],str3:my_sem_renorm[channels[0]],
                    str4:my_int_renorm[channels[1]],str5:my_sd_renorm[channels[1]],str6:my_sem_renorm[channels[1]],
                    str7:my_int_renorm[channels[2]],str8:my_sd_renorm[channels[2]],str9:my_sem_renorm[channels[2]]})
        dfout.to_csv(filename[:-4] + '.csv')
        
    plt.show()
    
    
def my_intensity_plot(int, row, **kwargs):
    """
    Plot track intensities from track array. 
    """
    n_channels = int.shape[-1]
    n_frames = int.shape[-2]
    int_c = np.moveaxis(int,-1,0)
        
    channels = kwargs.get('channels', [i for i in np.arange(n_channels)]) 
    xlim = kwargs.get('xlim', [0,0])
    ylim = kwargs.get('ylim', [0,0])
    colors = kwargs.get('colors', ['tab:red', 'tab:green', 'tab:blue','tab:orange','tab:purple'])
    markers = kwargs.get('markers', ['o','s','v','^','d'])
    filename = kwargs.get('filename', 'none')
    labels = kwargs.get('labels', ['ch1','ch2','ch3','ch4','ch5'])
    style = kwargs.get('style', 'seaborn-whitegrid')
    aspect_ratio = kwargs.get('aspect_ratio',0)
 
    plt.style.use(style)

    my_int = np.zeros([n_channels, n_frames])    
    fig, ax = plt.subplots()
    t = np.arange(n_frames)
    ch_index = 0
    for ch in channels:
        my_int[ch] = int_c[ch,row]
        ax.plot(t, my_int[ch], color=colors[ch_index], marker=markers[ch_index],label=labels[ch_index])
        ch_index = ch_index + 1
    ax.set(xlabel='frame #', ylabel='Intensity (a.u.)',
    title='Track Intensity vs. time')

    if xlim != [0,0]:
        plt.xlim(xlim[0], xlim[1])
    if ylim != [0,0]:
        plt.ylim(ylim[0], ylim[1])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if aspect_ratio != 0:
        ax.set_aspect(aspect_ratio) 
    if filename != 'none':
        plt.savefig(filename, format = 'svg', dpi=300)  
    plt.show()
 