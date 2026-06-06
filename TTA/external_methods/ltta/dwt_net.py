import torch.nn.functional as F
import torch
import torch.nn as nn
import pywt

def custom_prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row, device=None):
    
    g0_col, g1_col, g0_row, g1_row = torch.cat(g0_col), torch.cat(g1_col), torch.cat(g0_row), torch.cat(g1_row)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row

def custom_sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4

    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2 * lo.shape[d]
    
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    
    if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or mode == 'periodic' or mode == 1:
        pad = (L-2, 0) if d == 2 else (0, L-2)
        y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y

def custom_afb1d(x, h0, h1, mode='zero', dim=-1):
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]

    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    #print(h1.device)
    h = torch.cat([h0, h1] * C, dim=0).cuda()
    
    # Calculate the pad size
    outsize = int(N/L) #pywt.dwt_coeff_len(N, L, mode=mode)
    #print('outsize-> {} N-> {} L-> {}'.format(outsize, N, L))
    p = 2 * (outsize - 1) - N + L
    if mode == 0:
        if p % 2 == 1:
            pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
            x = F.pad(x, pad)
        pad = (p//2, 0) if d == 2 else (0, p//2)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return lohi

def custom_prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row, device=None):
    h0_col, h1_col, h0_row, h1_row = torch.cat(h0_col), torch.cat(h1_col), torch.cat(h0_row), torch.cat(h1_row)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        #J=1
        mode='zero'
        
        self.weights = nn.Parameter(torch.empty(1, 1))
        self.weights.data.fill_(0.7071067811865476)

    def dwt_forward(self, x):
        h0_col, h1_col = [self.weights, self.weights], [-self.weights, self.weights]
        h0_row, h1_row = h0_col, h1_col
        
        # Prepare the filters
        filts_forward = custom_prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row, device = self.weights.device)
        self.h0_col = filts_forward[0]
        self.h1_col = filts_forward[1]
        self.h0_row = filts_forward[2]
        self.h1_row = filts_forward[3]
        
        
        
        ll = x
        mode = 0
        lohi = custom_afb1d(x, self.h0_row, self.h1_row, mode=mode, dim=3)
        y = custom_afb1d(lohi, self.h0_col, self.h1_col, mode=mode, dim=2)

        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()

        return low, highs
    
    def dwt_inverse(self, yl, yh): 
        g0_col, g1_col = [self.weights, self.weights], [-self.weights, self.weights]
        g0_row, g1_row = g0_col, g1_col
        
        filts_inverse = custom_prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row, self.weights.device)
        self.g0_col =  filts_inverse[0]
        self.g1_col =  filts_inverse[1]
        self.g0_row =  filts_inverse[2]
        self.g1_row =  filts_inverse[3]
        
        
        ll = yl
        yh = [yh]
        mode = 1

        for h in yh[::-1]:
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]

            #ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            lh, hl, hh = torch.unbind(h, dim=2)
            lo = custom_sfb1d(ll, lh, self.g0_col, self.g1_col, mode=mode, dim=2)
            hi = custom_sfb1d(hl, hh, self.g0_col, self.g1_col, mode=mode, dim=2)
            ll = custom_sfb1d(lo, hi, self.g0_row, self.g1_row, mode=mode, dim=3)
        return ll
        
    def forward(self, x, xh=None):
        if xh == None:
            return self.dwt_forward(x)
        else:
            return self.dwt_inverse(x, xh)