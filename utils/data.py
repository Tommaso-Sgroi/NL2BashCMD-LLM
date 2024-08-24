from json import load

class NL2CMD:
    def __init__(self):
        self.invocation = ''
        self.cmd = ''


def get_dataset(path):
    with open(path, 'r') as f:
        dataset = load(f)
    return dataset