import sys
sys.path.append("/home/aa/as13000/FJX/Back_PSF/copy_psflearning/learning/fitters")
sys.path.append("/home/aa/as13000/FJX/Back_PSF/copy_psflearning/learning")
print(sys.path)
from FitterInterface_file import FitterInterface
from forwardProject import forward_project
from loss_functions import mse_real

from ast import Raise
import numpy as np
from numpy.core.fromnumeric import var
import scipy as sp
import tensorflow as tf
#from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

class Fitter(FitterInterface):
    
    def __init__(self, object, img, psf, optimizer = None, loss_func = None) -> None:
        self.psf = psf
        self.object = object
        self.img = img
        self.optimizer = optimizer
        self.loss_func = loss_func
        return 
    
    def learn_psf(self, variables: list = None) -> list:
        calculate_result = np.zeros((self.object.shape[0], self.object.shape[1], self.psf.shape[-1]))
        for u in range(self.psf.shape[-1]):
            calculate_result[:, :, u] = forward_project(self.psf[:, :, :, u], self.object)
        loss_calculate = mse_real(calculate_result,self.img)
        return super().learn_psf(variables)