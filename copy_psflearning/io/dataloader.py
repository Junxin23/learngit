from pickle import FALSE
import h5py as h5
import czifile as czi
import numpy as np
from skimage import io
# append the path of the parent directory as long as it's not a real package
import glob
import json
from PIL import Image
import scipy.io as scio


class Dataloader:
    def __init__(self,param=None):
        self.param = param

    def getfilelist(self, file_path):
        filelist = glob.glob(file_path+'/squareMLA_psf_sim'+'*'+'.mat')
        return filelist
    
    
    def loadimag(self,filename):
        #param = self.param
        dat = np.squeeze(io.imread(filename).astype(np.float32))
        #dat = (dat-param.ccd_offset)*param.gain
        return dat
    
    def loadobject(self, filename):
        #print(filename)
        dat = np.squeeze(io.imread(filename).astype(np.float32))
        #dat =scio.loadmat(filename).astype(np.float32)
        return dat
    
    def loadpsf(self, filelist):
        #dat = np.squeeze(io.imread(filename).astype(np.float32))
        #dat = scio.loadmat(filename).astype(np.float32)
        tmp = np.zeros((13, 13, 101, 377, 377))
        z = 0
        for filename in filelist:
            dat = h5.File(filename,'r')
            data = np.array(dat.get('psf_z'))
            tmp[:,:,z, :, :] = data
            z = z + 1
        return tmp