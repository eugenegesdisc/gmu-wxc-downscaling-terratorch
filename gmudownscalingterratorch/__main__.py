"""Command-line interace to GMU Downscaling TerraTorch"""

import sys
import logging
from huggingface_hub import hf_hub_download
from gmudownscalingterratorch.util.cli_utils import (
    gmu_build_lightning_cli
)
from .cli.gmu_command_main import (
    gmu_cmd_main
)
def main():
    if len(sys.argv) == 1:
        print("Usage: gmuldownscalingterratouch -h (cmd,fit,validate,test,predict) [-c CONFIG] [--print-config[=flags]] ...")
        exit(0)
    elif sys.argv[1] == "cmd":
        del sys.argv[1]
        gmu_cmd_main(prog_name="gmuldownscalingterratouch cmd")
    else:
        _ = gmu_build_lightning_cli()

if __name__ == "__main__":
    main()