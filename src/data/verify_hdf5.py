import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json

def validate_fusion_data(file_path):
    """
    验证 HDF5 文件中物理信号长度与元数据(Meta)的逻辑一致性
    输入: 托卡马克放电 HDF5 文件路径
    输出: 逻辑检查结果字典
    """
    with h5py.File(file_path, 'r') as f:
        # 1. 提取元数据 (Metadata)
        meta = f['meta']
        t_start = meta['StartTime'][()]  # 单位: s
        t_stop = meta['DownTime'][()]   # 单位: s
        dt = meta['time_interval'][()]   # 单位: s (例如 0.001 代表 1kHz)
        expected_len = meta['length'][()] # meta组记录的声明长度
        
        # 2. 提取实际数据组 (Data)
        # 以等离子体电流 ip 为例进行对齐检查
        actual_data = f['data/ip'][()]
        actual_len = len(actual_data)
        
        # 3. 逻辑计算
        # 计算理论持续时间点数 (考虑浮点数精度，使用 round)
        calculated_len = int(round((t_stop - t_start) / dt))
        
        # 4. 结果映射
        report = {
            "炮号": Path(file_path).stem,
            "StartTime": float(t_start),  # 转换为普通float避免numpy类型显示问题
            "DownTime": float(t_stop),
            "理论计算点数": calculated_len,
            "Meta声明长度": int(expected_len),  # 转换为普通int
            "Data实际点数": actual_len,
            "是否对齐": bool(actual_len == expected_len == calculated_len)  # 转换为普通bool
        }
        
        return report

def print_report_beautifully(report):
    """
    美观地打印报告结果
    """
    print("="*50)
    print("📊 EAST托卡马克数据验证报告")
    print("="*50)
    
    # 使用格式化输出避免字典默认显示问题
    print(f"🎯 炮号: {report['炮号']}")
    print(f"🕐 开始时间: {report['StartTime']:.6f} 秒")
    print(f"🏁 结束时间: {report['DownTime']:.6f} 秒")
    print(f"🔢 理论计算点数: {report['理论计算点数']}")
    print(f"📝 Meta声明长度: {report['Meta声明长度']}")
    print(f"📊 Data实际点数: {report['Data实际点数']}")
    print(f"✅ 是否对齐: {'是' if report['是否对齐'] else '否'}")
    
    print("="*50)

# 执行验证
result = validate_fusion_data("G:\\我的云端硬盘\\Fuison\\data\\EAST\\unified_hdf5\\54100\\54157.hdf5")

# 方法1: 美观打印
print_report_beautifully(result)

# 方法2: JSON格式化输出（避免编码问题）
print("\n📄 JSON格式输出:")
print(json.dumps(result, ensure_ascii=False, indent=2))

# 方法3: 如果需要原始字典显示，可以这样处理
print(f"\n🔍 原始字典内容:")
for key, value in result.items():
    print(f"  {key}: {value}")