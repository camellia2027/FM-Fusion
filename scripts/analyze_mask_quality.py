#!/usr/bin/env python3

import cv2
import numpy as np
import json
import sys

def analyze_mask_and_json(mask_path, json_path):
    """分析掩码图像和JSON的匹配情况"""
    
    print(f"=== 分析 {mask_path} 和 {json_path} ===")
    
    # 读取掩码图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取掩码图像: {mask_path}")
        return
    
    print(f"掩码图像尺寸: {mask.shape}")
    
    # 分析掩码中的唯一值
    unique_values = np.unique(mask)
    print(f"掩码中的唯一值: {unique_values}")
    
    # 统计每个值的像素数量
    for val in unique_values:
        count = np.sum(mask == val)
        print(f"  值 {val}: {count} 像素")
    
    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        print(f"JSON结构: {list(json_data.keys())}")
        
        if 'mask' in json_data:
            mask_entries = json_data['mask']
            print(f"JSON中的检测数量: {len(mask_entries)}")
            
            json_values = []
            for entry in mask_entries:
                if 'value' in entry:
                    val = entry['value']
                    json_values.append(val)
                    label = entry.get('label', entry.get('labels', 'unknown'))
                    print(f"  JSON值 {val}: {label}")
            
            # 检查匹配情况
            json_set = set(json_values)
            mask_set = set(unique_values.tolist())
            
            print(f"\n=== 匹配分析 ===")
            print(f"JSON中的值: {sorted(json_set)}")
            print(f"掩码中的值: {sorted(mask_set)}")
            
            missing_in_mask = json_set - mask_set
            missing_in_json = mask_set - json_set
            
            if missing_in_mask:
                print(f"JSON中有但掩码中没有的值: {missing_in_mask}")
            if missing_in_json:
                print(f"掩码中有但JSON中没有的值: {missing_in_json}")
            
            if not missing_in_mask and not missing_in_json:
                print("✅ JSON和掩码完全匹配！")
            else:
                print("❌ JSON和掩码不匹配！")
                
    except Exception as e:
        print(f"读取JSON文件失败: {e}")

def main():
    if len(sys.argv) != 3:
        print("用法: python3 analyze_mask_quality.py <mask_path> <json_path>")
        return
    
    mask_path = sys.argv[1]
    json_path = sys.argv[2]
    
    analyze_mask_and_json(mask_path, json_path)

if __name__ == '__main__':
    main()
