from resdata.grid import Grid
import numpy as np
import scipy.optimize
import warnings


# Mapping function for local cell corner indices
_ORIENT = {'NW': (0, 1), 'NE': (1, 1), 'SW': (0, 0), 'SE': (1, 0)}


class GridTool:
    def __init__(self, filename, fixed_layer_thickness=None):
        # Load grdecl file
        self.grid = Grid.load_from_grdecl(filename)

        # Load ZCORN; shape = (nz, 2, ny, 2, nx, 2), where 0 and 1 in the 2-dimension ndices will give you top/bottom,
        # north/south and east/west, respectively
        self.zcorn = np.reshape(self.grid.export_zcorn().numpy_copy(), 
                                shape=(self.grid.nz, 2, self.grid.ny, 2, self.grid.nx, 2))

        # Load COORD; shape = (ny + 1, nx + 1, 2, 3) with pillars starting in SW corner, the 2-dimenion index is
        # top/bottom coordinates given by 3-dimension index
        self.coord = np.reshape(self.grid.export_coord().numpy_copy(), 
                                shape=(self.grid.ny + 1, self.grid.nx + 1, 2, 3))

        # Store fixed layer thickness
        self.layer_thick = fixed_layer_thickness

    def move_zcorn_layer(self, k, h, redistribute=False):
        """ 
        Move corners in a k-layer up/down according to a vector input h.
        NOTE: if there is a fixed layer thickness, h must be shape = (nx, 2, ny, 2), otherwise shape = (2, nx, 2, ny, 2)
        """
        # Add h to zcorn at layer k, but ensure that it does not exceed original grid bounds
        tiled_zcorn_top = np.tile(self.zcorn[0, 0, :, :, :, :], (2, 1, 1, 1, 1))
        tiled_zcorn_bot = np.tile(self.zcorn[self.nz - 1, 1, :, :, :, :], (2, 1, 1, 1, 1))
        
        if self.layer_thick is None:
            # If h is constant, and tile it to match zcorn shape
            if h.ndim == 0:
                h = np.tile(h, self.zcorn[k, :, :, :, :, :].shape)

            # Find cells that exceed top/bottom bounds
            top_grid_prob = self.zcorn[k, :, :, :, :, :] + h < tiled_zcorn_top
            bot_grid_prob = self.zcorn[k, :, :, :, :, :] + h > tiled_zcorn_bot

            # Set exceeding cells to bounding zcorn values
            if np.any(top_grid_prob):
                self.zcorn[k, top_grid_prob] = tiled_zcorn_top[top_grid_prob]
            if np.any(bot_grid_prob):
                self.zcorn[k, bot_grid_prob] = tiled_zcorn_bot[bot_grid_prob]
            
            # Change cells with h for cells that have not exceeded bounds
            no_prob = np.logical_and(~top_grid_prob, ~bot_grid_prob)
            self.zcorn[k, no_prob] += h[no_prob]
            
        else:
            # If h is constant, and tile it to match zcorn shape
            if h.ndim == 0:
                h = np.tile(h, self.zcorn[k, 0, :, :, :, :].shape)

            # Find cells that exceed top/bottom bounds.
            top_grid_prob = self.zcorn[k, 0, :, :, :, :] + h < tiled_zcorn_top[0]
            bot_grid_prob_0 = self.zcorn[k, 0, :, :, :, : ] + h > tiled_zcorn_bot[0]
            bot_grid_prob_1 = self.zcorn[k, 0, :, :, :, :] + h + self.layer_thick > tiled_zcorn_bot[0]

            #Set exceeding cells to bounding zcorn values
            if np.any(top_grid_prob):
                self.zcorn[k, 0, top_grid_prob] = tiled_zcorn_top[0, top_grid_prob]
            if np.any(bot_grid_prob_0):
                self.zcorn[k, 0, bot_grid_prob_0] = tiled_zcorn_bot[0, bot_grid_prob_0]
            if np.any(bot_grid_prob_1):
                self.zcorn[k, 1, bot_grid_prob_1] = tiled_zcorn_bot[0, bot_grid_prob_1]
            
            #Change cells with h and layer_thick (constant) for cells that have not exceeded bounds
            no_prob_0 = np.logical_and(~top_grid_prob, ~bot_grid_prob_0)
            self.zcorn[k, 0, no_prob_0] += h[no_prob_0]
            self.zcorn[k, 1, ~bot_grid_prob_1] = self.zcorn[k, 0, ~bot_grid_prob_1] + self.layer_thick

        # Redistribute zcorns above and below layer k
        if redistribute:
            # Layers over layer k
            h_over = (self.zcorn[0, 0, :, :, :, :] - self.zcorn[k, 0, :, :, :, :]) / k

            # Loop over layers and add h_over to zcorn at top of layer k
            for i, layer in enumerate(range(k - 1, -1, -1)):
                self.zcorn[layer, 0, :, :, :, :] = self.zcorn[k, 0, :, :, :, :] + (i + 1) * h_over
                self.zcorn[layer, 1, :, :, :, :] = self.zcorn[k, 0, :, :, :, :] + i * h_over

            # Layers below layer k
            h_bel = (self.zcorn[self.nz - 1, 1, :, :, :, :] - self.zcorn[k, 1, :, :, :, :]) / (self.nz - k - 1)

            # Loop over layers below and add h_bel
            for i, layer in enumerate(range(k + 1, self.nz)):
                self.zcorn[layer, 0, :, :, :, :] = self.zcorn[k, 1, :, :, :, :] + i * h_bel
                self.zcorn[layer, 1, :, :, :, :] = self.zcorn[k, 1, :, :, :, :] + (i + 1) * h_bel

        # Move top and bottom of layers k + 1 and k - 1 to align with layer k
        else:
            # Additionally, move bottom corners in layer above (k - 1) and top corners in layer below (k + 1)
            self.zcorn[k - 1, 1, :, :, :, :] = self.zcorn[k, 0, :, :, :, :]
            self.zcorn[k + 1, 0, :, :, :, :] = self.zcorn[k, 1, :, :, :, :]
            
            # Check if adding h leads to problems
            top_prob = self.zcorn[k - 1, 1, :, :, :, :] < self.zcorn[k - 1, 0, :, :, :, :]
            bot_prob = self.zcorn[k + 1, 0, :, :, :, :] > self.zcorn[k + 1, 1, :, :, :, :]

            # # Set problematic zcorn to bounds
            if np.any(top_prob):
                warnings.warn(f'Top ZCORN values in layer {k} intersects with top values in {k - 1}. '
                            'Adjusting problematic ZCORNS values to coincide!')
                self.zcorn[k, 0, top_prob] = self.zcorn[k - 1, 0, top_prob]
                self.zcorn[k - 1, 1, top_prob] = self.zcorn[k - 1, 0, top_prob]
            if np.any(bot_prob):
                warnings.warn(f'Bottom ZCORN values in layer {k} intersects with bottom values in {k + 1}. '
                            'Adjusting problematic ZCORNS values to coincide!')
                self.zcorn[k, 1, bot_prob] = self.zcorn[k + 1, 1, bot_prob]
                self.zcorn[k + 1, 0, bot_prob] = self.zcorn[k + 1, 1, bot_prob]

    def move_zcorn_single(self, i, j, k, h, orient, bottom):
        """
        Move a zcorn up/down a pillar, corresponding to negative/positive h. 

        "orient" is given in compass directions, with the following logic:
        
            NW --- NE    j
            |       |    ^
            |       |    |
            SW --- SE     ---> i
        
        In some manuals, south and north are referred to as near and far, and west and east are referred to as left and
        right.

        "bottom" is a boolean indicating top (= 0 or False) or bottom (= 1 or True) cell surface
        """
        ind = _ORIENT[orient]
        self.zcorn[k, bottom, j, ind[1], i, ind[0]] += h
    
    def zcorn_layer_distance_simple(self, zcorn_pillar, k, tile_to_bottom=False):
        """
        Get the distance between incoming and existing zcorn at the top corners in a layer k.
        NOTE: We set all corner points connected to that pillar equal the value in zcorn_pillar, which should have shape
        = (nx + 1, ny + 1). Depending on tile_to_bottom we copy the h to bottom zcorns in layer k, which means that the
        original thickness at each pillar is kept when moved up or down
        """
        # Tile zcorn_pillar such that we can take the difference with zcorn directly
        pillar_rshp = np.ones((self.zcorn[k, 0, :, :, :, :].shape))
        for i in range(self.nx):
            for j in range(self.ny):
                pillar_rshp[j, 0, i, 0] = zcorn_pillar[j, i]
                pillar_rshp[j, 0, i, 1] = zcorn_pillar[j, i + 1]
                pillar_rshp[j, 1, i, 0] = zcorn_pillar[j + 1, i]
                pillar_rshp[j, 1, i, 1] = zcorn_pillar[j + 1, i + 1]
        
        # Calculate difference with zcorn
        h = pillar_rshp - self.zcorn[k, 0, :, :, :, :]
        if tile_to_bottom:
            h = np.tile(h, (2, 1, 1, 1, 1))
        return h

    def pillar_intersection(self, surf, ip, jp):
        """
        Find the z-coordinate where a pillar intersects with surface represented by surf=surf(x,y).
        NOTE: (ip, jp) are pillar indices which starts from SW corner of cell (0, 0)
        """
        # Top/bottom coordinates of pillar
        x1 = self.coord[jp, ip, 0, 0]
        x2 = self.coord[jp, ip, 1, 0]
        y1 = self.coord[jp, ip, 0, 1]
        y2 = self.coord[jp, ip, 1, 1]
        z1 = self.coord[jp, ip, 0, 2]
        z2 = self.coord[jp, ip, 1, 2]

        # If pillar is vertical, the intersection is trivial
        if x1 == x2 and y1 == y2:
            return surf(x1, y1)
        
        # Need intersection function for root
        def zcoord(r):
            if x1 != x2:
                return z1 + ((z2 - z1) / (x2 - x1)) * (r[0] - x1)
            else:
                return z1 + ((z2 - z1) / (y2 - y1)) * (r[1] - y1)
        def f(r):
            return zcoord(r) - surf(r[0], r[1])
        
        # Use scipy.optimize.root to calculate at which r=(x,y) the intersection occurs
        xmid = (x1 + x2) / 2
        ymid = (y1 + y2) / 2
        sol = scipy.optimize.minimize(f, x0=[xmid, ymid])

        return surf(sol.x[0], sol.x[1])
    
    def get_zcorn(self):
        return self.zcorn
    
    def set_zcorn(self, zcorn):
        self.zcorn = zcorn

    def get_coord(self):
        return self.coord
    
    def set_coord(self, coord):
        self.coord = coord
    
    @property
    def nx(self):
        return self.grid.nx
    
    @property
    def ny(self):
        return self.grid.ny
    
    @property
    def nz(self):
        return self.grid.nz
