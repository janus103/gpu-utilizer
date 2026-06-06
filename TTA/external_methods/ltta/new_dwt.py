import torch
import torch.nn as nn
import torch.nn.functional as F

class DWTForward(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters matching the DWT operation (need to be adjusted if the operation definition changes)
        self.ll_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]]).view(1, 1, 2, 2)
        self.hl_lh_hh_filters = torch.tensor([[[0.25, -0.25], [0.25, -0.25]], 
                                              [[0.25, 0.25], [-0.25, -0.25]], 
                                              [[0.25, -0.25], [-0.25, 0.25]]]).view(3, 1, 2, 2)

    def forward(self, input_data):
        channels = input_data.size(1)
        # Expand filters to match input channels
        ll_filter_expanded = self.ll_filter.repeat(channels, 1, 1, 1)
        hl_lh_hh_filters_expanded = self.hl_lh_hh_filters.repeat_interleave(channels, dim=0)
        
        # Applying filters
        LL = F.conv2d(input_data, ll_filter_expanded, groups=channels, stride=2)
        LH_HL_HH = F.conv2d(input_data.repeat(1, 3, 1, 1), hl_lh_hh_filters_expanded, groups=channels * 3, stride=2)
        
        # Extracting individual components
        LH, HL, HH = LH_HL_HH.chunk(3, dim=1)
        return LL, LH, HL, HH

class DWTInverse(nn.Module):
    def __init__(self):
        super().__init__()
        # 정의되는 커널은 역방향 변환에 맞게 설계되어야 합니다.
        # 각 커널은 역변환 시 적용되는 특정 서브밴드에 대응됩니다.
        self.ll_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]]).view(1, 1, 2, 2)
        self.lh_filter = torch.tensor([[0.25, -0.25], [0.25, -0.25]]).view(1, 1, 2, 2)
        self.hl_filter = torch.tensor([[0.25, 0.25], [-0.25, -0.25]]).view(1, 1, 2, 2)
        self.hh_filter = torch.tensor([[0.25, -0.25], [-0.25, 0.25]]).view(1, 1, 2, 2)

    def forward(self, LL, LH, HL, HH):
        channels = LL.size(1)
        # 필터 확장: 입력 채널 수에 맞게 각 필터를 확장합니다.
        ll_filter_expanded = self.ll_filter.repeat(channels, 1, 1, 1)
        lh_filter_expanded = self.lh_filter.repeat(channels, 1, 1, 1)
        hl_filter_expanded = self.hl_filter.repeat(channels, 1, 1, 1)
        hh_filter_expanded = self.hh_filter.repeat(channels, 1, 1, 1)

        # 역 컨볼루션 연산 적용
        LL_recon = F.conv_transpose2d(LL, ll_filter_expanded, stride=2, groups=channels)
        LH_recon = F.conv_transpose2d(LH, lh_filter_expanded, stride=2, groups=channels)
        HL_recon = F.conv_transpose2d(HL, hl_filter_expanded, stride=2, groups=channels)
        HH_recon = F.conv_transpose2d(HH, hh_filter_expanded, stride=2, groups=channels)

        # 각 결과를 결합하여 최종 이미지 재구성
        reconstructed = LL_recon + LH_recon + HL_recon + HH_recon
        return reconstructed

# Define the original functions without optimization
def dwt_2d_forward_torch(input_data):
    batch_size, channels, rows, cols = input_data.shape
    LL = torch.zeros((batch_size, channels, rows // 2, cols // 2))
    LH = torch.zeros((batch_size, channels, rows // 2, cols // 2))
    HL = torch.zeros((batch_size, channels, rows // 2, cols // 2))
    HH = torch.zeros((batch_size, channels, rows // 2, cols // 2))

    for b in range(batch_size):
        for c in range(channels):
            for i in range(0, rows, 2):
                for j in range(0, cols, 2):
                    LL[b, c, i // 2, j // 2] = (input_data[b, c, i, j] + input_data[b, c, i, j + 1] + input_data[b, c, i + 1, j] + input_data[b, c, i + 1, j + 1]) / 4
                    LH[b, c, i // 2, j // 2] = (input_data[b, c, i, j] - input_data[b, c, i, j + 1] + input_data[b, c, i + 1, j] - input_data[b, c, i + 1, j + 1]) / 4
                    HL[b, c, i // 2, j // 2] = (input_data[b, c, i, j] + input_data[b, c, i, j + 1] - input_data[b, c, i + 1, j] - input_data[b, c, i + 1, j + 1]) / 4
                    HH[b, c, i // 2, j // 2] = (input_data[b, c, i, j] - input_data[b, c, i, j + 1] - input_data[b, c, i + 1, j] + input_data[b, c, i + 1, j + 1]) / 4

    return LL, LH, HL, HH

def dwt_2d_inverse_torch(LL, LH, HL, HH):
    batch_size, channels, rows, cols = LL.shape
    reconstructed = torch.zeros((batch_size, channels, rows * 2, cols * 2))

    for b in range(batch_size):
        for c in range(channels):
            for i in range(rows):
                for j in range(cols):
                    reconstructed[b, c, 2 * i, 2 * j] = LL[b, c, i, j] + LH[b, c, i, j] + HL[b, c, i, j] + HH[b, c, i, j]
                    reconstructed[b, c, 2 * i, 2 * j + 1] = LL[b, c, i, j] - LH[b, c, i, j] + HL[b, c, i, j] - HH[b, c, i, j]
                    reconstructed[b, c, 2 * i + 1, 2 * j] = LL[b, c, i, j] + LH[b, c, i, j] - HL[b, c, i, j] - HH[b, c, i, j]
                    reconstructed[b, c, 2 * i + 1, 2 * j + 1] = LL[b, c, i, j] - LH[b, c, i, j] - HL[b, c, i, j] + HH[b, c, i, j]

    return reconstructed / 4

# Define the optimized forward and inverse functions
def dwt_2d_forward_torch_optimized(input_data, use_freq=[True, True, True, True]):
    batch_size, channels, rows, cols = input_data.shape
    # print(f'LOG (3) {batch_size} {channels} {rows} {cols}')
    LL = (input_data[..., ::2, ::2] + input_data[..., ::2, 1::2] + input_data[..., 1::2, ::2] + input_data[..., 1::2, 1::2]) / 4
    LH = (input_data[..., ::2, ::2] - input_data[..., ::2, 1::2] + input_data[..., 1::2, ::2] - input_data[..., 1::2, 1::2]) / 4
    HL = (input_data[..., ::2, ::2] + input_data[..., ::2, 1::2] - input_data[..., 1::2, ::2] - input_data[..., 1::2, 1::2]) / 4
    HH = (input_data[..., ::2, ::2] - input_data[..., ::2, 1::2] - input_data[..., 1::2, ::2] + input_data[..., 1::2, 1::2]) / 4

    if use_freq[0] == False:
        LL = torch.zeros_like(LL)
    if use_freq[1] == False:
        LH = torch.zeros_like(LH)
    if use_freq[2] == False:
        HL = torch.zeros_like(HL)
    if use_freq[3] == False:
        HH = torch.zeros_like(HH)

    return LL, LH, HL, HH

def dwt_2d_inverse_torch_optimized(LL, LH, HL, HH):
    batch_size, channels, rows, cols = LL.shape
    height, width = rows * 2, cols * 2
    reconstructed = torch.zeros((batch_size, channels, height, width), device=LL.device)
    reconstructed[..., ::2, ::2] = LL + LH + HL + HH
    reconstructed[..., ::2, 1::2] = LL - LH + HL - HH
    reconstructed[..., 1::2, ::2] = LL + LH - HL - HH
    reconstructed[..., 1::2, 1::2] = LL - LH - HL + HH
    return reconstructed / 4

def verify_with_conv():
    # Initialize sample input data
    input_data = torch.randn(1, 1, 8, 8)

    # Original functions application
    LL_orig, LH_orig, HL_orig, HH_orig = dwt_2d_forward_torch_optimized(input_data)
    reconstructed_orig = dwt_2d_inverse_torch_optimized(LL_orig, LH_orig, HL_orig, HH_orig)

    # New module instantiation and application
    dwt_forward = DWTForward()
    dwt_inverse = DWTInverse()

    LL_new, LH_new, HL_new, HH_new = dwt_forward(input_data)
    reconstructed_new = dwt_inverse(LL_new, LH_new, HL_new, HH_new)

    # Comparison
    print("Forward Comparison:", torch.allclose(LL_orig, LL_new) and torch.allclose(LH_orig, LH_new) and torch.allclose(HL_orig, HL_new) and torch.allclose(HH_orig, HH_new))
    print("Inverse Comparison:", torch.allclose(reconstructed_orig, reconstructed_new))

def verify():
    # Test to verify the equality of outputs from both implementations
    input_data = torch.rand(2, 3, 8, 8)  # Example input

    # Apply original and optimized DWT followed by IDWT
    LL, LH, HL, HH = dwt_2d_forward_torch(input_data)
    reconstructed_original = dwt_2d_inverse_torch(LL, LH, HL, HH)

    LL_opt, LH_opt, HL_opt, HH_opt = dwt_2d_forward_torch_optimized(input_data)
    reconstructed_optimized = dwt_2d_inverse_torch_optimized(LL_opt, LH_opt, HL_opt, HH_opt)

    # Check if the results are the same
    # print(torch.eq(reconstructed_original, reconstructed_optimized), "The results do not match!")
    print(torch.eq(LL, LL_opt), "The results do not match!")
    print(torch.eq(LH, LH_opt), "The results do not match!")
    print(torch.eq(HL, HL_opt), "The results do not match!")
    print(torch.eq(HH, HH_opt), "The results do not match!")

    print(f'LL. shape {LL.shape}')
    print(f'LH. shape {LH.shape}')
    print(f'HL. shape {HL.shape}')
    print(f'HH. shape {HH.shape}')

    print(f'Original {input_data}')

    print(f'LL.  {LL}')
    print(f'LH.  {LH}')
    print(f'HL.  {HL}')
    print(f'HH.  {HH}')

