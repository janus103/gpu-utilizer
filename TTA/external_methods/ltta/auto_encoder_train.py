#!/usr/bin/env python3
""" Model Inspection Script
Derived from validate_dwt.py
"""
import argparse
import os
import logging
import torch
import torch.nn as nn
from timm.models import create_model, load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('inspect')

parser = argparse.ArgumentParser(description='PyTorch Model Inspection')
# Keep arguments from original script to ensure compatibility with the provided command
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend")
parser.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry')
parser.add_argument('--dwt-kernel-size', nargs='*', type=int, default=[0, 0, 0],
                    help='dwt kernel size')
parser.add_argument('--dwt_bn', nargs='*', type=int, default=[0,0,0],
                    help='0: BatchNorm2D, 1: IBN, 2: IW')
parser.add_argument('--dwt_level', nargs='*', type=int, default=[2,2,2],
                    help='DWT Level on Layers')
parser.add_argument('--sfn', default='No-Names', type=str,
                    help='test name ')
parser.add_argument('--corrupted', default='None', type=str,
                    help='')
parser.add_argument('--img-mode', default='RGB', type=str,
                    help='')
parser.add_argument('--deep-format', action='store_true', default=False,
                    help='Learnable Deep Format')
parser.add_argument('--ena-dwt-ratio', action='store_true', default=False,
                    help='Learnable Deep Format')
parser.add_argument('--dwt_quant', type=int, default=1,
                    help='dwt_quantization')
parser.add_argument('--drop_low', action='store_true', default=False,
                    help='drop out only Low frequency ')
parser.add_argument('--vit', action='store_true', default=False,
                    help='w\0 DWT check ')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels')
parser.add_argument('--post-dwt', action='store_true', default=False,
                    help='post-dwt you must catch in-chans')
parser.add_argument('--aux-header', action='store_true', default=False,
                    help='add auxilary header layer')
parser.add_argument('--no-skip', action='store_true', default=False,
                    help='Do not identity mapping')
parser.add_argument('--dataset-alias', default='imagenet', type=str,
                    help='alias of dataset')
parser.add_argument('--weight_net', type=float, default=0., metavar='N',
                    help='weight of frequency weight net ')
parser.add_argument('--meta_option', type=int, default=0, metavar='N',
                    help='Meta Option for AdaIN Network')

# AutoEncoder arguments
parser.add_argument('--squeeze-dim', type=int, default=128,
                    help='Dimension of the latent vector for AutoEncoder')
parser.add_argument('--ae-epochs', type=int, default=1000,
                    help='Number of epochs for AutoEncoder training')
parser.add_argument('--ae-lr', type=float, default=0.01,
                    help='Learning rate for AutoEncoder')
parser.add_argument('--ae-loss', type=str, default='mse', choices=['mse', 'mae', 'combined'],
                    help='Loss function for AutoEncoder: mse, mae, or combined (mse + mae)')

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            # nn.Sigmoid() # Not using Sigmoid as input is not normalized to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    args = parser.parse_args()

    # Model loading logic
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    _logger.info(f"Creating model {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        aux_header = args.aux_header,
        no_skip = args.no_skip,
        dwt_kernel_size = args.dwt_kernel_size,
        dwt_level = args.dwt_level,
        dwt_bn = args.dwt_bn,
        deep_format = args.deep_format,
        meta_option=args.meta_option
    )

    if args.checkpoint:
        _logger.info(f"Loading checkpoint {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, args.use_ema)

    # 1. Find first convolutional layer
    first_conv = None
    # We iterate modules to find the first Conv2d
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            _logger.info(f"Found first Conv2d layer: {name}")
            break
    
    if first_conv is None:
        _logger.error("No Conv2d layer found in the model.")
        return

    # 2. Check for weights and bias
    has_weight = first_conv.weight is not None
    has_bias = first_conv.bias is not None
    
    _logger.info(f"Has weight: {has_weight}")
    _logger.info(f"Has bias: {has_bias}")

    # 3. Combine weight and bias into one tensor and print sizes
    if has_weight:
        weight = first_conv.weight.data
        print(f"Weight kernel size: {weight.shape}")

        # Calculate Mean and Variance for each kernel
        # weight shape: [out_channels, in_channels, k, k] -> [64, 3, 7, 7]
        # Flatten spatial dimensions: [64, 3, 49]
        weight_spatial_flat = weight.view(weight.size(0), weight.size(1), -1)
        
        # Calculate mean and variance along the spatial dimension (dim=2)
        # Result shape: [64, 3, 1]
        kernel_means = weight_spatial_flat.mean(dim=2, keepdim=True)
        kernel_vars = weight_spatial_flat.var(dim=2, keepdim=True)
        
        print(f"Kernel Means shape (before reshape): {kernel_means.shape}")
        print(f"Kernel Vars shape (before reshape): {kernel_vars.shape}")
        
        # Reshape to 1D tensor [64*3*1]
        kernel_means_1d = kernel_means.view(-1)
        kernel_vars_1d = kernel_vars.view(-1)
        
        print(f"Kernel Means 1D shape: {kernel_means_1d.shape}")
        print(f"Kernel Vars 1D shape: {kernel_vars_1d.shape}")
        
        # 1. Concatenate mean and var
        # [192] + [192] -> [384]
        kernel_concat_1d = torch.cat([kernel_means_1d, kernel_vars_1d], dim=0)
        print(f"Kernel Concat 1D shape: {kernel_concat_1d.shape}")

        # 2. Setup AutoEncoder
        input_dim = kernel_concat_1d.shape[0] # Should be 384
        latent_dim = args.squeeze_dim
        
        ae_model = AutoEncoder(input_dim, latent_dim).to(args.device)
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.ae_lr)
        
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        
        # Prepare input
        # Add batch dimension: [384] -> [1, 384]
        input_tensor = kernel_concat_1d.unsqueeze(0).to(args.device)
        
        _logger.info(f"Starting AutoEncoder training: {input_dim} -> {latent_dim} -> {input_dim} with {args.ae_loss} loss")
        
        # Setup for saving best model
        if not os.path.exists('output'):
            os.makedirs('output')
        save_path = f'output/ae_{latent_dim}.pth.tar'
        best_mse = float('inf')
        
        # 3. Training Loop
        ae_model.train()
        for epoch in range(args.ae_epochs):
            # Forward
            output = ae_model(input_tensor)
            
            # Calculate individual losses for monitoring/saving
            current_mse = mse_criterion(output, input_tensor)
            current_mae = mae_criterion(output, input_tensor)
            
            if args.ae_loss == 'mse':
                loss = current_mse
            elif args.ae_loss == 'mae':
                loss = current_mae
            elif args.ae_loss == 'combined':
                loss = current_mse + current_mae
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save best model based on MSE
            if current_mse.item() < best_mse:
                best_mse = current_mse.item()
                torch.save(ae_model.state_dict(), save_path)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.ae_epochs}], Loss: {loss.item():.6f}, MSE: {current_mse.item():.6f}")

        print("Training finished.")
        print(f"Best MSE: {best_mse:.8f}")
        
        # Load best model for validation
        if os.path.exists(save_path):
            print(f"Loading best model from {save_path}")
            ae_model.load_state_dict(torch.load(save_path))
        
        # 4. Validation / Inspection
        ae_model.eval()
        with torch.no_grad():
            reconstructed = ae_model(input_tensor)
            
            # Compare first 10 values
            print("\n--- Validation Results ---")
            print("Original (first 10):", input_tensor[0, :10].cpu().numpy())
            print("Reconstructed (first 10):", reconstructed[0, :10].cpu().numpy())
            
            # Calculate errors
            mse = nn.functional.mse_loss(reconstructed, input_tensor).item()
            mae = nn.functional.l1_loss(reconstructed, input_tensor).item()
            max_error = torch.max(torch.abs(reconstructed - input_tensor)).item()
            
            print(f"MSE: {mse:.8f}")
            print(f"MAE: {mae:.8f}")
            print(f"Max Absolute Error: {max_error:.8f}")
            print("--------------------------\n")

        # Flatten weight
        weight_flat = weight.view(-1)
        
        if has_bias:
            bias = first_conv.bias.data
            print(f"Bias size: {bias.shape}")
            bias_flat = bias.view(-1)
            # Concatenate
            combined = torch.cat([weight_flat, bias_flat])
        else:
            print("Bias size: None")
            combined = weight_flat
            
        print(f"Combined tensor shape: {combined.shape}")
        # print(combined) # Optional: print the actual tensor if needed, but might be large

if __name__ == '__main__':
    main()
