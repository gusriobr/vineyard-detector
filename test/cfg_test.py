import os

TEST_BASE = os.path.dirname(os.path.abspath(__file__))


def resource(filename):
    return os.path.join(TEST_BASE, "resources", filename)

def results(filename):
    return os.path.join(TEST_BASE, "results", filename)
