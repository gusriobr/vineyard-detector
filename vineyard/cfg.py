import logging
import os
import sys
from pathlib import Path

PROJECT_BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()


def resource(filename):
    return os.path.join(PROJECT_BASE, "resources", filename)


def results(filename):
    return os.path.join(PROJECT_BASE, "results", filename)


def project_file(filename):
    return os.path.join(PROJECT_BASE, filename)


LOG_CONFIGURATION = True


def configLog(level=logging.INFO):
    global LOG_CONFIGURATION
    if LOG_CONFIGURATION:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-1s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.FileHandler("vineyard.log"),
                                logging.StreamHandler()
                            ])
        # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        LOG_CONFIGURATION = True
