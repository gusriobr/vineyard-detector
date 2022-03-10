import os
from pathlib import Path

PROJECT_BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()


def resource(filename):
    return os.path.join(PROJECT_BASE, "resources", filename)

def results(filename):
    return os.path.join(PROJECT_BASE, "results", filename)
