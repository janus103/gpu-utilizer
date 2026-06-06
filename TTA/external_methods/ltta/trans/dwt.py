import numpy as np
from PIL import Image
import pywt

const_dwt_method = 'db1'

def dwt(ori_np):
    coeffs = pywt.dwt2(ori_np,const_dwt_method)
    return coeffs

def remove_high_freq(ori_np, lev:int):
    r_LL, (_,_,_) = dwt(ori_np[:,:,0])
    g_LL, (_,_,_) = dwt(ori_np[:,:,1])
    b_LL, (_,_,_) = dwt(ori_np[:,:,2])

    zero = np.zeros(r_LL.shape)

    rc = (r_LL, (zero, zero, zero))
    gc = (g_LL, (zero, zero, zero))
    bc = (b_LL, (zero, zero, zero))

    rec_rc = pywt.idwt2(rc, const_dwt_method)
    rec_gc = pywt.idwt2(gc, const_dwt_method)
    rec_bc = pywt.idwt2(bc, const_dwt_method)
    sum_np = np.zeros((rec_bc.shape[0],rec_bc.shape[1],3), dtype = np.uint8)
    sum_np[:,:,0] = rec_rc
    sum_np[:,:,1] = rec_gc
    sum_np[:,:,2] = rec_bc
    return sum_np, rec_rc, rec_gc, rec_bc

def get_freqs(ori_np,channel):
    assert (channel < 3), f'Channel is not 3 {ori_np.shape}'
    LL, (LH,HL,HH) = dwt(ori_np[:,:,channel])
    
    LL = LL.astype(np.uint8)
    HL = HL.astype(np.uint8)
    LH = LH.astype(np.uint8)
    HH = HH.astype(np.uint8)
    
    return LL,LH,HL,HH

def get_freqs_1(ori_np):
    #assert (channel < 3), f'Channel is not 3 {ori_np.shape}'
    LL, (LH,HL,HH) = dwt(ori_np)
    
    LL = LL.astype(np.uint8)
    HL = HL.astype(np.uint8)
    LH = LH.astype(np.uint8)
    HH = HH.astype(np.uint8)
    
    return LL,LH,HL,HH
def get_freqs_channel_1(ori_np):
    #assert (channel < 3), f'Channel is not 3 {ori_np.shape}'
    
    LL_Y, LH_Y, HL_Y, HH_Y = get_freqs_1(ori_np,0)
    #LL_U, LH_U, HL_U, HH_U = get_freqs(ori_np,1)
    #LL_V, LH_V, HL_V, HH_V = get_freqs(ori_np,2)
    #print('Y -> ',np.shape(ori_np))
    #print('Y_DWT -> ',np.shape(LL_Y))
    sum_np_LL = np.zeros((LL_Y.shape[0], LL_Y.shape[1], 1), dtype = np.uint8)
    sum_np_LH = np.zeros((LH_Y.shape[0], LH_Y.shape[1], 1), dtype = np.uint8)
    sum_np_HL = np.zeros((HL_Y.shape[0], HL_Y.shape[1], 1), dtype = np.uint8)
    sum_np_HH = np.zeros((HH_Y.shape[0], HH_Y.shape[1], 1), dtype = np.uint8)
    
    sum_np_LL[:,:,0] = LL_Y
    #sum_np_LL[:,:,1] = LL_U
    #sum_np_LL[:,:,2] = LL_V
    
    sum_np_LH[:,:,0] = LH_Y
    #sum_np_LH[:,:,1] = LH_U
    #sum_np_LH[:,:,2] = LH_V
    
    sum_np_HL[:,:,0] = HL_Y
    #sum_np_HL[:,:,1] = HL_U
    #sum_np_HL[:,:,2] = HL_V
    
    sum_np_HH[:,:,0] = HH_Y
    #sum_np_HH[:,:,1] = HH_U
    #sum_np_HH[:,:,2] = HH_V
    
    return sum_np_LL, sum_np_LH, sum_np_HL, sum_np_HH

def get_freqs_channel_3(ori_np):
    #assert (channel < 3), f'Channel is not 3 {ori_np.shape}'
    
    LL_Y, LH_Y, HL_Y, HH_Y = get_freqs(ori_np,0)
    LL_U, LH_U, HL_U, HH_U = get_freqs(ori_np,1)
    LL_V, LH_V, HL_V, HH_V = get_freqs(ori_np,2)
    #print('Y -> ',np.shape(ori_np))
    #print('Y_DWT -> ',np.shape(LL_Y))
    sum_np_LL = np.zeros((LL_Y.shape[0], LL_Y.shape[1],3), dtype = np.uint8)
    sum_np_LH = np.zeros((LH_Y.shape[0], LH_Y.shape[1],3), dtype = np.uint8)
    sum_np_HL = np.zeros((HL_Y.shape[0], HL_Y.shape[1],3), dtype = np.uint8)
    sum_np_HH = np.zeros((HH_Y.shape[0], HH_Y.shape[1],3), dtype = np.uint8)
    
    sum_np_LL[:,:,0] = LL_Y
    sum_np_LL[:,:,1] = LL_U
    sum_np_LL[:,:,2] = LL_V
    
    sum_np_LH[:,:,0] = LH_Y
    sum_np_LH[:,:,1] = LH_U
    sum_np_LH[:,:,2] = LH_V
    
    sum_np_HL[:,:,0] = HL_Y
    sum_np_HL[:,:,1] = HL_U
    sum_np_HL[:,:,2] = HL_V
    
    sum_np_HH[:,:,0] = HH_Y
    sum_np_HH[:,:,1] = HH_U
    sum_np_HH[:,:,2] = HH_V
    
    return sum_np_LL, sum_np_LH, sum_np_HL, sum_np_HH

def get_dwt_components(original_img, img_size=224):
    original_img = original_img.resize((img_size,img_size))
    np_original_img = np.array(original_img)
    #print('xxxx')
    LL, LH, HL, HH = get_freqs_channel_3(np_original_img)
    LL1, LH1, HL1, HH1 = get_freqs_channel_3(LL)
    LL2, LH2, HL2, HH2 = get_freqs_channel_3(LH)
    LL3, LH3, HL3, HH3 = get_freqs_channel_3(HL)
    LL4, LH4, HL4, HH4 = get_freqs_channel_3(HH)
    total_sum = np.sum((LL1, LH1, HL1, HH1,\
                       LL2, LH2, HL2, HH2,\
                       LL3, LH3, HL3, HH3,\
                       LL4, LH4, HL4, HH4))
    #print(f'-------- {np.sum(LL1)} {total_sum} {np.sum(LL1)/total_sum} ')
    #{np.sum(LL1)} {total_sum} {np.sum(LL1)/total_sum}
    np.sum(LL1)/total_sum
    return [np.sum(LL1)/total_sum, np.sum(LH1)/total_sum, np.sum(HL1)/total_sum, np.sum(HH1)/total_sum,\
           np.sum(LL2)/total_sum, np.sum(LH2)/total_sum, np.sum(HL2)/total_sum, np.sum(HH2)/total_sum,\
           np.sum(LL3)/total_sum, np.sum(LH3)/total_sum, np.sum(HL3)/total_sum, np.sum(HH3)/total_sum,\
           np.sum(LL4)/total_sum, np.sum(LH4)/total_sum, np.sum(HL4)/total_sum, np.sum(HH4)/total_sum]