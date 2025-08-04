"""
计算CIF文件的perplexity值。
Perplexity是语言模型评估序列预测能力的指标，数值越低表示模型对该序列的预测能力越好。
"""
import os
import math
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from pymatgen.core.structure import Structure
from crystallm import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
)

from bin.prepare_csv_benchmark import process_cif_files
from bin.tar_to_pickle import load_data_from_tar, save_data_to_pickle


@dataclass
class NLLDefaults:
    out_dir: str = "crystallm_v1_small"  # 包含训练模型的目录路径
    input_file: str = "/data/wuhl/CrysBFN/bfn_samples.csv"
    # input_file: str = "/data/wuhl/CrysBFN/remote/DiffCSP/diffcsp_samples.csv"
    # input_file: str = "resources/benchmarks/mp_20/train.csv"  # 输入文件路径
    device: str = "cuda"  
    dtype: str = "float32"  
    compile: bool = False  
    chunk_size: int = 1024  
    num_workers: int = 4  
    csv_column: str = "cif"  


if __name__ == "__main__":
    # 解析配置
    C = parse_config(NLLDefaults)
    
    # 验证必要参数
    if not C.input_file:
        print("错误: 必须指定input_file参数")
        exit(1)
    
    if not os.path.exists(C.input_file):
        raise FileNotFoundError(f"输入文件不存在: {C.input_file}")
    
    # 设置设备和数据类型
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in C.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 处理CSV文件
    print(f"读取生成样本CSV文件: {C.input_file}")
    df = pd.read_csv(C.input_file)
    
    if C.csv_column not in df.columns:
        raise ValueError(f"CSV文件中不存在列 '{C.csv_column}'")
    
    # Step 1: 准备数据
    df = df.head(1000)
    if 'material_id' not in df.columns:
        df['material_id'] = [f'mp_gen_{i}' for i in range(len(df))]
    
    # 把要处理的df复制并保存在 ./my_tmp/
    tmp_path = 'my_tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    tmp_csv = os.path.join(tmp_path, 'tmp.csv')
    df.to_csv(tmp_csv, index=False)
    
    processed_cif_path = os.path.join(tmp_path, 'tmp.tar.gz')
    process_cif_files(input_csv=tmp_csv, output_tar_gz=processed_cif_path)
    cif_data = load_data_from_tar(processed_cif_path)
    pkl_path = os.path.join(tmp_path, 'tmp.pkl.gz')
    save_data_to_pickle(cif_data, pkl_path)
    
    prep_pkl_path = os.path.join(tmp_path, 'tmp_prep.pkl.gz')
    cmd_str = f"python bin/preprocess.py {pkl_path} --out {prep_pkl_path}"
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise Exception(f"Preprocess failed with exit code {exit_code}")
    
    # Step 2: Tokenize
    cmd_str = f"python bin/tokenize_cifs.py --train_fname {prep_pkl_path} --val_fname {prep_pkl_path} --out_dir {tmp_path} --workers 16"
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise Exception(f"Tokenize failed with exit code {exit_code}")
    
    # step 3, 运行 NLL 计算
    cmd_str = f"python bin/calc_nll.py --config config/crystallm_mp_20_small_ppl.yaml device=cuda dtype=float16 dataset={tmp_path} out_dir={tmp_path}"
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise Exception(f"Prepare train failed with exit code {exit_code}")
        

