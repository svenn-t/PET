import argparse
from pipt import pipt_init
from pipt.loop.assimilation import Assimilate
from input_output import read_config
from importlib import import_module


def main():
    # Setup argparse
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument('init_file', type=str, help='init file for PIPT')
    
    # Optional arguments
    parser.add_argument('-s', '--sim', type=str, dest='sim', help='simulator to use in PIPT', required=True)

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
