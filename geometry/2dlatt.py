from abc import ABC, abstractmethod
import os
import sys

import base

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import *

def rotation(rad:float)-> np.ndarray:
    '''
    Create a 2D rotation matrix for a given angle in radians.
    Args:
        rad: The angle in radians to rotate.
    Returns:
        A 2x2 numpy array representing the rotation matrix.
    '''
    return np.array([[np.cos(rad), -np.sin(rad)],
                     [np.sin(rad),  np.cos(rad)]])
