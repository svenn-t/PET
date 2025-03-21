""" Simple class to make EGRID files using resdata """
from resdata.grid import Grid
from scipy.spatial import KDTree
import numpy as np
import os

from simulator.eclipse import eclipse

class egrid(eclipse):
    """
    Open a GRDECL file and write a EGRID
    """
    def __init__(self, input_dict=None, filename=None, options=None):
        # Call eclipse.__init__
        super().__init__(input_dict, filename, options)

        # Internal variables
        self.tree = None

        # egrid info in input dict
        self._ext_info_input_dict()

    def _ext_info_input_dict(self):
        # Make a KDTree with info from input egrid file
        if 'kdtree' in self.input_dict:
            # Instantiate Grid
            grid = Grid(self.input_dict['kdtree'][0])
            
            # Get k-layers in input grid
            self.kdtree_k_layer = self.input_dict['kdtree'][1]
            assert len(self.kdtree_k_layer) == len(self.k_layer)

            # Set up on tree per layer
            self.tree = [None for _ in self.kdtree_k_layer]
            for ind, k in enumerate(self.kdtree_k_layer):
                coord = np.array([np.mean(grid[i, j, k].corners[:4], axis=0) for i in range(grid.nx) for j in 
                                  range(grid.ny) if grid.active(ijk=(i, j, k))])
                self.tree[ind] = KDTree(coord)

    def call_sim(self, folder=None, wait_for_proc=False):
        # Filename
        if folder is not None:
            filename = folder + self.file
        else:
            filename = self.file
        
        # Load GRDECL file and save to EGRID
        grid = Grid.load_from_grdecl(f'{filename}.GRDECL')
        grid.save_EGRID(f'{filename}.EGRID')

        return True
    
    def get_sim_results(self, whichResponse, ext_data_info=None, member=None):
        # 
        # Layer top surface
        #
        if whichResponse == 'surface' or whichResponse == 'mean_surface':
            # self.tree cannot be None
            assert self.tree is not None, 'KDTREE keyword must be given in input file for \"surface\" data type!'

            # File name depends on if member is None
            filename = f'{self.file}.EGRID'
            if member is not None:
                filename = f'En_{member}{os.sep}{filename}'

            # Only instantiate new Grid object if necessary
            if hasattr(self, 'kdtree_grid'):
                # Doublecheck if path is correct, and if not instantiate correct grid
                rt_mem = int(self.kdtree_grid.get_name().split('/')[0].split('_')[1])
                if rt_mem != member:
                    self.kdtree_grid = Grid(f'{filename}')
            else:
                self.kdtree_grid = Grid(f'{filename}')
            
            # Extract top surface coordinates
            k = int(ext_data_info[1]) - 1
            coord = np.array([np.mean(self.kdtree_grid[i, j, k].corners[:4], axis=0) for i in 
                              range(self.kdtree_grid.nx) for j in range(self.kdtree_grid.ny) if 
                              self.kdtree_grid.active(ijk=(i, j, k))])

            # Query stored KDTree to get distance between surfaces
            ind = self.k_layer.index(k)
            y_egrid, _ = self.tree[ind].query(coord, p=1)
            if whichResponse == 'mean_surface':
                y_egrid = np.array([np.mean(y_egrid)])

        return y_egrid
