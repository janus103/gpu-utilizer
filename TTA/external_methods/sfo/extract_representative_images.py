#!/usr/bin/env python3
"""
대표 이미지 추출 스크립트 (Representative Image Extractor)

기능:
1. Source Dataset의 모든 이미지를 Pretrained Model에 통과시켜 Feature를 추출합니다.
2. 각 클래스별로 Feature들의 평균(Centroid)을 계산합니다.
3. Centroid와 가장 거리가 가까운 상위 K개의 이미지를 '대표 이미지'로 선정합니다.
4. 선정된 이미지를 Target Dataset 경로로 구조를 유지하며 복사합니다.

사용법:
python extract_representative_images.py \
  --source /data/jin/imagenet/val \
  --target /data/jin/imagenet_1shot_val \
  --samples-per-class 1 \
  --model resnet50 \
  --pretrained \
  --batch-size 256
"""

import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Representative Image Extractor using Pretrained Features')
    parser.add_argument('--source', type=str, required=True, help='원본 데이터셋 경로 (ImageFolder 구조, validation set 권장)')
    parser.add_argument('--target', type=str, required=True, help='생성될 데이터셋 경로')
    parser.add_argument('--samples-per-class', type=int, default=1, help='클래스 당 추출할 대표 이미지 수')
    parser.add_argument('--model', type=str, default='resnet50', help='Feature Extractor 모델명 (timm)')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--pretrained-path', default=None, type=str, help='Load this checkpoint as if they were the pretrained weights')
    parser.add_argument('--batch-size', type=int, default=128, help='Inference 배치 사이즈')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader 워커 수')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    return parser.parse_args()

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, pretrained=False, pretrained_path=None):
        super().__init__()
        
        # Pretrained Config 설정
        pretrained_cfg_overlay = None
        if pretrained_path:
            pretrained_cfg_overlay = dict(file=pretrained_path)
            # pretrained=True로 설정해야 create_model 내부에서 가중치를 로드함 (timm 로직)
            pretrained = True 
            
        # Pretrained 모델 로드 (Classifier 제외)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, # Feature Extraction Mode (No Classifier)
            pretrained_cfg_overlay=pretrained_cfg_overlay
        )
        self.model.eval()
        
        # 모델의 입력 크기 등 설정 가져오기
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        
    def forward(self, x):
        # Forward pass features (Global Pooled)
        return self.model(x)

def main():
    args = parse_args()
    
    # 1. 설정 및 준비
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    # 2. 모델 로드
    print(f"Loading model: {args.model} (Pretrained: {args.pretrained})...")
    if args.pretrained_path:
        print(f"Loading custom checkpoint: {args.pretrained_path}")
        
    extractor = FeatureExtractor(args.model, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    
    # Transform 설정 (모델에 맞는 정규화 적용)
    # Validation 모드로 transform 생성 (Resize + CenterCrop + Normalize)
    transforms_list = timm.data.create_transform(
        **timm.data.resolve_data_config(extractor.data_config, model=extractor.model),
        is_training=False
    )
    
    print(f"Data Transform: {transforms_list}")
    
    # 3. 데이터셋 로드
    dataset = datasets.ImageFolder(args.source, transform=transforms_list)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=args.num_workers, pin_memory=True)
    
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes.")
    
    # 4. Feature Extraction
    print("Extracting features...")
    features_dict = defaultdict(list) # {class_idx: [feature_vector]}
    indices_dict = defaultdict(list)  # {class_idx: [dataset_index]}
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            outputs = extractor(images) # [B, Feature_Dim]
            
            # CPU로 이동하여 저장 (메모리 관리)
            outputs = outputs.cpu().numpy()
            targets = targets.numpy()
            
            for j in range(len(targets)):
                class_idx = targets[j]
                global_idx = i * args.batch_size + j
                features_dict[class_idx].append(outputs[j])
                indices_dict[class_idx].append(global_idx)
    
    # 5. 대표 이미지 선정 및 복사
    print(f"Selecting {args.samples_per_class} representative images per class...")
    
    if os.path.exists(args.target):
        print(f"Warning: Target directory '{args.target}' already exists. Merging/Overwriting...")
        os.makedirs(args.target, exist_ok=True)
    else:
        os.makedirs(args.target)
        
    copy_count = 0
    skipped_classes = 0
    
    # 클래스 인덱스를 이름으로 변환하기 위한 매핑
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    # 진행 상황 표시를 위해 정렬된 키 사용
    sorted_class_indices = sorted(features_dict.keys())
    
    for class_idx in tqdm(sorted_class_indices):
        # 해당 클래스의 모든 Feature
        class_features = np.array(features_dict[class_idx]) # [N, Dim]
        class_indices = indices_dict[class_idx]
        
        if len(class_features) == 0:
            skipped_classes += 1
            continue
            
        # 1) Centroid (평균) 계산
        centroid = np.mean(class_features, axis=0) # [Dim]
        
        # 2) Centroid와의 거리(L2) 계산
        distances = np.linalg.norm(class_features - centroid, axis=1) # [N]
        
        # 3) 거리가 가장 가까운 상위 K개 선택
        # argsort는 오름차순 정렬 (작은 값이 먼저)
        num_select = min(len(distances), args.samples_per_class)
        selected_local_indices = np.argsort(distances)[:num_select]
        
        # 4) 파일 복사
        class_name = idx_to_class[class_idx]
        target_class_dir = os.path.join(args.target, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        
        for local_idx in selected_local_indices:
            global_idx = class_indices[local_idx]
            original_path = dataset.samples[global_idx][0]
            filename = os.path.basename(original_path)
            
            target_path = os.path.join(target_class_dir, filename)
            
            # 파일 복사
            try:
                shutil.copy2(original_path, target_path)
                copy_count += 1
            except Exception as e:
                print(f"Failed to copy {original_path} to {target_path}: {e}")
            
    print(f"\nProcessing Complete.")
    print(f" - Source: {args.source}")
    print(f" - Target: {args.target}")
    print(f" - Images copied: {copy_count}")
    if skipped_classes > 0:
        print(f" - Skipped classes (empty): {skipped_classes}")

if __name__ == '__main__':
    main()
