"""
    The main command line program for pygeoutil.
"""
import logging
import argparse
from . import gmu_plugin_command_manager


def gmu_parse_args(prog_name="python gmu_command_main.py"):
    """
        Parsing the arguments.
    """
    parser = argparse.ArgumentParser(
        prog=prog_name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Utility programs for preparing and pre-processing data for deep learning. 
""",
        epilog=f"""
Examples:
    {prog_name} --help 
"""
    )

    # options applicable to all sub-commands
    parser.add_argument(
        '--debug', action="store_true", required=False, default=False,
        help='Enable debug mode')
    
    parser.add_argument('--loglevel', default='ERROR',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (e.g., DEBUG, INFO)')
    # sub-command: enable sum-command
    subcmd_parsers = parser.add_subparsers(title="subcommands",
                                        description="A set of sub-commands",
                                        dest="subcmd",
                                        required=True,
                                        help="List of sub-commands")
    for plugin_name in gmu_plugin_command_manager.discovered_subcommand_plugins:
        the_plugin=gmu_plugin_command_manager.discovered_subcommand_plugins[plugin_name]
        the_plugin.add_cli_arg(subcmd_parsers)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    return args


def process_args(args):
    """
        Processing all the options of arguments.
    """
    if args.debug:
        print("args=", args)
    the_sub_cmd = "gmudownscalingterratorch.subcommands.gmu_downscaling_subcommand_"+args.subcmd
    if (the_sub_cmd in gmu_plugin_command_manager.discovered_subcommand_plugins):
        the_plugin = gmu_plugin_command_manager.discovered_subcommand_plugins.get(the_sub_cmd)
        if the_plugin is not None:
            the_plugin.process(args)

def gmu_cmd_main(prog_name="python -m gmuldownscalingterratouch"):
    args = gmu_parse_args(prog_name=prog_name)
    if args.debug:
        print("discovered_plugins=",gmu_plugin_command_manager.discovered_subcommand_plugins)
    process_args(args)

if __name__ == '__main__':
    gmu_cmd_main(prog_name="python gmudownscalingterratorch/util/gmu_command_main.py")