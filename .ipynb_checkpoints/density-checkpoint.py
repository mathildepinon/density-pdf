import os
import copy
import time
import sys

import numpy as np

from mockfactory import Catalog
from pypower import CatalogMesh
from pypower.utils import sky_to_cartesian
from utils import BaseClass


class Density(BaseClass):
    """
    Class to compute density field from a data catalog.
    """
    _defaults = dict(data_positions=None, position_type='xyz', data_weights=None, randoms_positions=None, randoms_weights=None, mesh=None, boxsize=None, boxcenter=None)
    
    def __init__(self, data_positions, **kwargs):
        """
        Initialize :class:`Density`.

        Parameters
        ----------
        data_positions : array of shape (3, N)
            Positions of the data points.

        position_type : str
            Format of the input positions, 'xyz' cartesian coordinates or 'rdd' for ra, dec, distance.

        data_weights : array, optional
            Weights of the data points. If not provided, all points are assumed to have the same weight.

        data_weights : array, optional
            Weights of the data points. If not provided, all points are assumed to have the same weight.

        randoms_positions : array, optional
            Positions of the random points. If not provided, boxsize must be provided.

        randoms_weights : array, optional
            Weights of the random points. If not provided, all points are assumed to have the same weight.

        boxsize : array of shape (3,), optional
            Size of the box. If not provided, boxsize is computed from randoms positions.

        boxcenter : array of shape (3,), optional, default=0
            Center of the box.
        """
        BaseClass.__init__(self, data_positions=data_positions, **kwargs)
        
        self.data_positions = np.asarray(data_positions)
        
        #if self.data_weights is None:
        #    self.data_weights = np.ones(len(data_positions))
  
        #if self.randoms_positions is not None and self.randoms_weights is None:
        #    self.randoms_weights = np.ones(len(randoms_positions))
       
    def compute_density(self, cellsize, resampler='tsc', smoothing_radius=None, use_weights=True, interlacing=0, min_ran=0, sampling=None, seed=0, sampling_size=5):

        data_positions = self.data_positions
        randoms_positions = self.randoms_positions

        if self.position_type == 'rdd':
            data_positions = np.array(sky_to_cartesian(self.data_positions))
            if self.randoms_positions is not None:
                randoms_positions = np.array(sky_to_cartesian(self.randoms_positions))
                
        print(data_positions.shape)
        
        mesh = CatalogMesh(data_positions=data_positions, randoms_positions=randoms_positions, 
                           data_weights=self.data_weights, randoms_weights=self.randoms_weights,
                           boxsize=self.boxsize, boxcenter=self.boxcenter,
                           cellsize=cellsize, interlacing=interlacing, resampler=resampler)
        
        nmesh = mesh.nmesh[0]
        self.mesh = mesh
        data_mesh = self.mesh.to_mesh(field='data')
        if smoothing_radius is not None:
            data_mesh = data_mesh.r2c().apply(TopHat(r=smoothing_radius))
            data_mesh = data_mesh.c2r()        
        self.data_mesh = data_mesh
        # Compute density contrast on a grid
        if randoms_positions is not None:
            randoms_mesh = self.mesh.to_mesh(field='data-normalized_randoms')
            if smoothing_radius is not None:
                randoms_mesh = randoms_mesh.r2c().apply(TopHat(r=smoothing_radius))
                randoms_mesh = randoms_mesh.c2r()
            self.randoms_mesh = randoms_mesh

            sum_data, sum_randoms = data_mesh.csum(), randoms_mesh.csum()
            density_mesh = data_mesh - randoms_mesh
            mask = randoms_mesh > min_ran
            density_mesh[mask] /= randoms_mesh[mask]
            density_mesh[~mask] = np.nan
        else:
            norm = len(self.data_positions[0]) if self.data_weights is None else np.sum(self.data_weights)
            density_mesh = data_mesh/(norm/nmesh**3) - 1
        
        self.density_mesh = density_mesh
          
        # Get densities at each point
        resampler_conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}
        
        if sampling == 'randoms':
            density = density_mesh.readout(randoms_positions.T, resampler=resampler_conversions[resampler])
        elif sampling == 'data':
            density = density_mesh.readout(data_positions.T, resampler=resampler_conversions[resampler])
        else:
            nrandoms = sampling_size * data_positions.shape[1]
            rng = np.random.RandomState(seed=seed)
            offset = self.mesh.boxcenter - self.mesh.boxsize / 2
            randoms = np.array([o + rng.uniform(0., 1., nrandoms)*b for o, b in zip(offset, self.mesh.boxsize)])
            density = density_mesh.readout(randoms.T, resampler=resampler_conversions[resampler])
            
        self.density = density 
        self.cellsize = cellsize
        self.resampler = resampler
        self.interlacing = interlacing
        
        
class TopHat(object):
    '''Top-hat filter in Fourier space
    adapted from https://github.com/bccp/nbodykit/

    Parameters
    ----------
    r : float
        the radius of the top-hat filter
    '''
    def __init__(self, r):
        self.r = r

    def __call__(self, k, v):
        r = self.r
        k = sum(ki ** 2 for ki in k) ** 0.5
        kr = k * r
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 3 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)
        w[k == 0] = 1.0
        return w * v
    
    
if __name__ == '__main__':
    from cosmoprimo.fiducial import DESI
    
    # Y1 blinded data
    data_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/blinded/LRG_SGC_clustering.dat.fits'
    randoms_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.1/blinded/LRG_SGC_{}_clustering.ran.fits'
    
    data = Catalog.read(data_fn)
    randoms = Catalog.concatenate([Catalog.read(randoms_fn.format(i)) for i in range(18)]) 
    
    data_positions = [data['RA'], data['DEC'], DESI().comoving_radial_distance(data['Z'])]
    randoms_positions = [randoms['RA'], randoms['DEC'], DESI().comoving_radial_distance(randoms['Z'])]
    
    density = Density(data_positions=data_positions, randoms_positions=randoms_positions, position_type='rdd', data_weights=data['WEIGHT'], randoms_weights=randoms['WEIGHT'])
    
    t0 = time.time()
    
    density.compute_density(smoothing_radius=20, cellsize=10, min_ran=0.1, sampling='randoms')
    
    np.save('/global/cfs/cdirs/desi/users/mpinon/density/y1/blinded/v1.1/LRG_SGC_density_r20_cellsize10_nran18_minran0.1_inputrandomssampling', density.density)
    
    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
