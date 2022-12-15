import argparse
from pipt import pipt_init
from pipt.loop.assimilation import Assimilate
from input_output import read_config
from importlib import import_module
import numpy as np
import pickle


def main():
    # Setup argparse
    parser = argparse.ArgumentParser(
        description='Run PIPT using specifications in the init_file and simulator defined by -s (or --sim).', 
        usage='%(prog)s [options] init_file -s simulator',
        formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=35))

    # Positional arguments
    parser.add_argument('init_file', type=str, help='init file for PIPT')
    
    # --sim is really a positional argument
    group = parser.add_argument_group('required')
    group.add_argument('-s', '--sim', type=str, metavar='', dest='sim', 
        help='simulator to use in PIPT', required=True)

    # Optional arguments
    group2 = parser.add_argument_group('options')
    group2.add_argument('--save', type=str, default='state', choices=['state'],
        help='what results to save (default = state)')
    group2.add_argument('--save-name', type=str, metavar='', default='pipt_results',
        help='results save name (default = \"pipt_results\")')
    group2.add_argument('--save-fmt', type=str, default='numpy', choices=['numpy', 'pickle'],
        help='serialization method (default = numpy)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Read config
    if args.init_file.endswith('.pipt'):
        da, fwd = read_config.read_txt(args.init_file)
    elif args.init_file.endswith('.toml'):
        da, fwd = read_config.read_toml(args.init_file)
    elif args.init_file.endswith('.yaml'):
        da, fwd = read_config.read_yaml(args.init_file)
    else:
        raise TypeError(f'\"{args.init_file.split(".")[-1]}\" is not a valid init_file format')

    # Import simulator class
    sim_class = getattr(import_module('simulator.' + '.'.join(args.sim.split('.')[:-1])), args.sim.split('.')[-1])
    sim = sim_class(fwd)

    # Instantiate Assimilation
    analysis = pipt_init.init_da(da, fwd, sim)
    assimilation = Assimilate(analysis)

    # Run assimilation
    assimilation.run()

    # Serialize results according to cli args
    savename = args.save_name
    if args.save_fmt == 'numpy':
        if args.save == 'state':
            np.savez(savename + '.npz', **assimilation.ensemble.state)
    elif args.save_fmt == 'pickle':
        with open(savename + '.p', 'wb') as fid:
            if args.save == 'state':
                pickle.dump(assimilation.ensemble.state, fid)
