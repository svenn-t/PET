""" Simple class to make EGRID files using resdata """
from resdata.grid import Grid
import cwrap

from simulator.eclipse import eclipse

class egrid(eclipse):
    """
    Open a GRDECL file and write a EGRID
    """
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
        