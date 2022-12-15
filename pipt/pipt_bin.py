import argparse
from pipt import pipt_init
from pipt.loop.assimilation import Assimilate
from input_output import read_config
from importlib import import_module


def main():
    # Setup argparse
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument('init_file', type=str, help='init file (.pipt) for PIPT')
    
    # Optional arguments
    parser.add_argument('-s', '--sim', type=str, dest='sim', help='simulator to use in PIPT', required=True)

    # Parse command-line arguments
    args = parser.parse_args()

    # Read config
    da, fwd = read_config.read_txt(args.init_file)

    # Import simulator class
    sim_class = getattr(import_module('simulator.' + '.'.join(args.sim.split('.')[:-1])), args.sim.split('.')[-1])
    sim = sim_class(fwd)

    # Instantiate Assimilation
    analysis = pipt_init.init_da(da, fwd, sim)
    assimilation = Assimilate(analysis)

    # Run assimilation
    assimilation.run()
