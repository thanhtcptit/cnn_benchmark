import argparse

from nsds.common import Params
from nsds.common.util import import_submodules
from nsds.commands.subcommand import ArgumentParserWithDefaults, Subcommand

from cnn_benchmark.train import run_training
from cnn_benchmark.benchmark import run_benchmark

import_submodules('cnn_benchmark')


def main():
    parser = ArgumentParserWithDefaults()
    subparsers = parser.add_subparsers()

    SUBCOMMAND_COLLECTIONS = {
        'train': Train(),
        'benchmark': Benchmark()
    }

    for name, subcommand in SUBCOMMAND_COLLECTIONS.items():
        subparser = subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()
    args.func(args)


class Train(Subcommand):
    def add_subparser(self, name, subparsers):
        description = 'Train the specified model on the specified dataset.'
        subparser = subparsers.add_parser(name, description=description)

        subparser.add_argument(
            'param_path',
            type=str,
            help='path to parameter file describing the model to be trained')
        subparser.add_argument(
            '-c', '--checkpoint_dir',
            default=None,
            type=str,
            help=('directory in which to save the model and its logs.'
                  'Put None to name the directory based on current time'))
        subparser.add_argument(
            '-r', '--recover',
            action='store_true',
            default=False,
            help='recover training from the state in checkpoint_dir')
        subparser.add_argument(
            '-f', '--force',
            action='store_true',
            required=False,
            help='overwrite the output directory if it exists')

        subparser.set_defaults(func=parse_param_and_run_training)
        return subparser


def parse_param_and_run_training(args):
    params = Params.from_file(args.param_path)
    return run_training(params, args.checkpoint_dir, args.recover, args.force)


class Benchmark(Subcommand):
    def add_subparser(self, name, subparsers):
        description = 'Run benchmark.'
        subparser = subparsers.add_parser(name, description=description)

        subparser.add_argument(
            'param_path',
            type=str,
            help='path to parameter file describing the benchmark to be run')
        subparser.add_argument(
            '-c', '--checkpoint_dir',
            default=None,
            type=str,
            help=('directory in which to save the model and its logs.'
                  'Put None to name the directory based on current time'))
        subparser.add_argument(
            '--show_logs',
            type=int,
            choices=[0, 1],
            default=0,
            help=('Put 1 to send logs to stdout')
        )

        subparser.set_defaults(func=parse_param_and_run_benchmark)
        return subparser


def parse_param_and_run_benchmark(args):
    params = Params.from_file(args.param_path)
    return run_benchmark(params, args.checkpoint_dir, args.show_logs)


if __name__ == '__main__':
    main()
