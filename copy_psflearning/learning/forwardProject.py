import cupy as cp

def forward_project(psf, Xguess):
    ra, ca = Xguess.shape[0], Xguess.shape[1]
    rb, cb = psf.shape[0], psf.shape[1]
    
    r = ra + rb - 1
    c = ca + cb - 1
    
    a1 = cp.zeros((r,c), dtype=cp.float32)
    b1 = cp.zeros((r,c), dtype=cp.float32)
    con1 = cp.zeros((r,c), dtype=cp.float32)
    
    for z in range(Xguess.shape[2]):
        a1[:ra, :ca] = Xguess[:, :, z]
        b1[:rb, :cb] = psf[:, :, z]
        
        con1 = con1 + cp.fft.fft2(a1[:ra, :ca], s=(r, c)) * cp.fft.fft2(b1[:rb, :cb], s=(r, c))
        
    projection1 = cp.real(cp.fft.ifft2(con1))
    
    projection = projection1[round((r - ra) / 2):round((r - ra) / 2) + ra,
                             round((c - ca) / 2):round((c - ca) / 2) + ca]
    
    return projection