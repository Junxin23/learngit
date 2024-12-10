import tensorflow as tf

import numpy as np


def mse_real(model,data,variables=None,mu=None,w=None):
    mydiff = model-data
    mydiff = mydiff[:,1:-1]
    data = data[:,1:-1]
    model = model[:,1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data) 
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    #f = variables[3]  
    #gxymean = tf.reduce_mean(tf.abs(variables[4]))   
    #bg = variables[1]
    #intensity = variables[2]
    #s = tf.math.reduce_sum(tf.math.square(f[0]-f[1])+tf.math.square(f[-1]-f[-2]))
   
    #dfz = tf.math.square(tf.experimental.numpy.diff(f, n = 1, axis = -3))
    #dfz = tf.reduce_sum(dfz)
   
    #Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(f,0)))
    #bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    #intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    #fsz = f.shape
    #ccz = fsz[0]//2
    #wd = tf.math.minimum(cc,10)
    # g0 = f
    # g = g0[:,1:-1,1:-1]
    # Imin1 = tf.reduce_sum(tf.math.square(g0))-tf.reduce_sum(tf.math.square(g))
    
    # Inorm = tf.math.abs(tf.math.reduce_sum(f))
    #Inorm = tf.reduce_mean(tf.math.square(tf.math.reduce_sum(f,axis=(-1,-2))-tf.math.reduce_sum(f)/fsz[0]))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + w[2]*dfz + s*w[3] + w[4]*Imin*mu + bgmin*w[5]*mu  + intensitymin*w[6]*mu + Inorm*w[7] + gxymean*w[8]
    loss = mse_norm1*w[0] + mse_norm2*w[1]
    #loss = LL*w[0] + w[2]*dfz + s*w[3] + w[4]*Imin*mu + bgmin*w[5]*mu  + intensitymin*w[6]*mu + Inorm*w[7]*mu + gxymean*w[8]

    return loss



def mse_real_All(model,data,loss_func,variables=None,mu=None,w=None, psfnorm=None):
    varsize = len(variables)
    var = [None]*(varsize-1)
    loss = 0.0
    for i in range(0,model.shape[0]):
        for j in range(1,varsize-1):
            var[j] = variables[j][i]
        var[0] = variables[0]
        if psfnorm:
            loss += loss_func(model[i],data[i],var,mu,w,psfnorm[i])
        else:
            loss += loss_func(model[i],data[i],var,mu,w)
    
    return loss