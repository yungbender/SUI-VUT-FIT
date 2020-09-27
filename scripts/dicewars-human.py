#!/usr/bin/env python3
from signal import signal, SIGCHLD
from subprocess import Popen
from argparse import ArgumentParser

from utils import log_file_producer


parser = ArgumentParser(prog='Dice_Wars')
parser.add_argument('-b', '--board', help="Seed for generating board", type=int)
parser.add_argument('-s', '--strength', help="Seed for dice assignment", type=int)
parser.add_argument('-o', '--ownership', help="Seed for province assignment", type=int)
parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
parser.add_argument('-l', '--logdir', help="Folder to store last running logs in.")
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('--ai', help="Specify AI versions as a sequence of ints.", nargs='+')

procs = []


def signal_handler(signum, frame):
    """Handler for SIGCHLD signal that terminates server and clients
    """
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass


def main():
    """
    Run the Dice Wars game.

    Example:
        ./scripts/dicewars-human.py --ai dt.sdc dt.ste dt.stei dt.wpm_c
    """
    args = parser.parse_args()

    signal(SIGCHLD, signal_handler)

    if len(args.ai) > 7:
        print("Only games of up to 7 AI players are supported")
        exit(1)

    try:
        cmd = [
            "./scripts/server.py",
            "-n", str(len(args.ai) + 1),
            "-p", str(args.port),
            "-a", str(args.address),
            "--dice-assignment", 'flat',
        ]
        if args.board is not None:
            cmd.extend(['-b', str(args.board)])
        if args.ownership is not None:
            cmd.extend(['-o', str(args.ownership)])
        if args.strength is not None:
            cmd.extend(['-s', str(args.strength)])
        if args.debug:
            cmd.extend(['--debug', 'DEBUG'])

        procs.append(Popen(cmd, stderr=log_file_producer(args.logdir, 'server.log')))

        cmd = [
            "./scripts/client.py",
            "-p", str(args.port),
            "-a", str(args.address),
        ]
        if args.debug:
            cmd.extend(['--debug', 'DEBUG'])
        procs.append(Popen(cmd, stderr=log_file_producer(args.logdir, 'client-human.log')))

        for ai in args.ai:
            cmd = [
                "./scripts/client.py",
                "-p", str(args.port),
                "-a", str(args.address),
                "--ai", ai,
            ]
            if args.debug:
                cmd.extend(['--debug', 'DEBUG'])

            procs.append(Popen(cmd, stderr=log_file_producer(args.logdir, 'client-{}.log'.format(ai))))

        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        for p in procs:
            p.kill()


if __name__ == '__main__':
    main()
