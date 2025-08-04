# Optional: if you want to generate samples from CrysBFN and DiffCSP again
# 
# conda activate crysbfn
# cd /data/wuhl/CrysBFN
# bash scripts/gen_scripts/mp20_exps.sh
# bash scripts/gen_scripts/mp20_eval.sh

# cd /data/wuhl/DiffCSP
# python scripts/generation.py --model_path /data/wuhl/DiffCSP/mp_gen --dataset mp_20
# python scripts/compute_metrics.py --root_path /data/wuhl/DiffCSP/mp_gen --tasks gen --gt_file /data/wuhl/DiffCSP/data/mp_20/test.csv

# Run NLL calculation
conda activate crystallm
cd /data/wuhl/CrystaLLM
python bin/run_cal_nll_pipeline.py input_file=/data/wuhl/DiffCSP/diffcsp_samples.csv
python bin/run_cal_nll_pipeline.py input_file=/data/wuhl/CrysBFN/bfn_samples.csv
