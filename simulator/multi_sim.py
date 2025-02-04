"""" Generic classes for running multiple simulators """
import numpy as np


class sim_container:
    def __init__(self, sims, input_dict=None, filename=None, options=None):
        # Internalize list of simluators
        self.sims = sims

        # Internalize other input
        self.input_dict = input_dict
        self.file = filename
        self.options = options

    def setup_fwd_run(self, **kwargs):
        # Loop over simulators and run individual setup
        for sim in self.sims:
            sim.setup_fwd_run(**kwargs)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # Initialize output from simulators
        pred_data = []

        # Loop over simulators and run
        for sim in self.sims:
            pred_data.append(sim.run_fwd_sim(state, member_i, del_folder))

        # Convert pred. data to list and return
        self.pred_data = [{k: v for k, v in elem.items()} for elem in pred_data[0]]
        for sim_pd in pred_data[1:]:
            for i, pd in enumerate(sim_pd):
                for k, v in pd.items():
                    self.pred_data[i][k] = np.hstack((self.pred_data[i][k], v))

        return self.pred_data
    