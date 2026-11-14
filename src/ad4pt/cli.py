"""One command: preview strawberries and save counts when streaming stops."""

import argparse

from ad4pt.vision.demo import add_demo_command, demo_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live ripe/unripe strawberry counts")
    commands = parser.add_subparsers(dest="mode")
    demo = add_demo_command(commands)
    args = parser.parse_args(argv)
    if args.mode is None:
        args = demo.parse_args([])
    return demo_command(args)
