import h5py
import numpy as np

# 读取HDF5文件
file_path = r"G:\我的云端硬盘\Fuison\data\EAST\unified_hdf5\53800\53825.hdf5"

def print_structure(name, obj):
    """递归打印HDF5文件结构"""
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name}")
        print(f"{indent}   Shape: {obj.shape}")
        print(f"{indent}   Dtype: {obj.dtype}")
        # 打印属性
        if len(obj.attrs) > 0:
            print(f"{indent}   Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}     - {attr_name}: {attr_value}")

print("="*80)
print(f"HDF5 文件结构: {file_path}")
print("="*80)

with h5py.File(file_path, 'r') as f:
    # 打印文件级属性
    if len(f.attrs) > 0:
        print("\n📋 文件级属性:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  - {attr_name}: {attr_value}")
    
    print("\n📂 文件结构:")
    print("-"*80)
    
    # 递归遍历所有组和数据集
    f.visititems(print_structure)
    
    print("\n" + "="*80)
    print("详细数据组信息:")
    print("="*80)
    
    # 检查主要数据组
    for key in f.keys():
        print(f"\n🔹 主组: /{key}")
        group = f[key]
        if isinstance(group, h5py.Group):
            print(f"  包含的数据集:")
            for subkey in group.keys():
                dataset = group[subkey]
                if isinstance(dataset, h5py.Dataset):
                    print(f"    • {subkey}:")
                    print(f"      - Shape: {dataset.shape}")
                    print(f"      - Dtype: {dataset.dtype}")
                    print(f"      - Size: {dataset.size} elements")
                    # 如果数据较小，显示一些样本值
                    if dataset.size <= 10:
                        print(f"      - Values: {dataset[:]}")
                    else:
                        print(f"      - First 5 values: {dataset[:5]}")
                        print(f"      - Last 5 values: {dataset[-5:]}")
                    
                    # 显示属性
                    if len(dataset.attrs) > 0:
                        print(f"      - Attributes:")
                        for attr_name, attr_value in dataset.attrs.items():
                            print(f"        * {attr_name}: {attr_value}")
    
    # 特别检查 n=1 amplitude 数据
    print("\n" + "="*80)
    print("🎯 特别关注: n=1 amplitude 数据")
    print("="*80)
    
    # 尝试查找 n=1 amplitude 相关的数据
    possible_paths = [
        'data/n=1 amplitude',
        'CIII/n=1 amplitude',
        'n=1 amplitude',
        'data/n_1_amplitude',
        'CIII/n_1_amplitude'
    ]
    
    for path in possible_paths:
        if path in f:
            dataset = f[path]
            print(f"\n找到数据路径: {path}")
            print(f"  Shape: {dataset.shape}")
            print(f"  Dtype: {dataset.dtype}")
            print(f"  Min: {np.min(dataset[:])}")
            print(f"  Max: {np.max(dataset[:])}")
            print(f"  Mean: {np.mean(dataset[:])}")
            print(f"  Std: {np.std(dataset[:])}")
            break
