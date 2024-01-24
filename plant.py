import numpy as np
from helpers import noise


class Plant(object):
    def __init__(self):
        self.name = None

    def update(self, u):
        d = noise()
