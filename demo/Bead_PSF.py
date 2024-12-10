import sys
sys.path.append("/home/aa/as13000/FJX/Back_PSF")
import numpy as np
from copy_psflearning.learning.forwardProject import forward_project
from copy_psflearning.io.dataloader import Dataloader
from copy_psflearning.learning.fitters.Fitter_file import Fitter
# ToDO: take a look at the fitterinterface_file which can be used directly in this code.

#from copy_psflearning.io.makeplots import *
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except:
    print('Running on CPU')
    
#L = psflearninglib()
#L.param = io.param.combine('config_base',psftype='voxel',channeltype='1ch',sysfile='M2')

##Loading Part
Path_of_Object = '/home/aa/as13000/FJX/3D_deconvolution/FJX/code/256-768.tif'
Path_of_Image = '/home/aa/as13000/FJX/3D_deconvolution/FJX/code/save_path/inp_1.tif'
Path_of_PSf = "/home/aa/as13000/FJX/3D_deconvolution/FJX/code/PSF/psfmatrix_1.4NA_zspacing0.1/squareMLA_H.mat"

Dataloader = Dataloader()
object_Raw = Dataloader.loadobject(Path_of_Object)
Image_Raw = Dataloader.loadimag(Path_of_Image)

Image = Image_Raw.transpose(1, 2, 0) #changed it into (x,y, angle=169) 
object = object_Raw.transpose(1, 2, 0) #changed it into (x,y, angle=169) 

PSF_Filelist = Dataloader.getfilelist("/home/aa/as13000/FJX/3D_deconvolution/FJX/code/PSF/psfmatrix_1.4NA_zspacing0.1/")
PSF = Dataloader.loadpsf(PSF_Filelist)

PSF_Reshape = np.reshape(PSF, (PSF.shape[0] * PSF.shape[1], PSF.shape[2], PSF.shape[3], PSF.shape[4]))
PSF_Reshape = np.transpose(PSF_Reshape, (2, 3, 1, 0))

wdf = np.zeros((object.shape[0], object.shape[1], PSF_Reshape.shape[-1]))
for u in range(PSF_Reshape.shape[-1]):
    wdf[:, :, u] = forward_project(PSF_Reshape[:, :, :, u], object)
    
PSF_init = np.zeros((PSF.shape[0] * PSF.shape[1], PSF.shape[2], PSF.shape[3], PSF.shape[4]))

Optimizer = Fitter(object, wdf, PSF_Reshape)
Optimizer.learn_psf()