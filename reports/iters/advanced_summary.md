# 高级模型汇总

- 总运行数：`3`
- 推荐运行：`adv_mamba_lite_ws128_st16_e5_s42`

| run_name | model_name（模型名称） | test_acc | test_auc | test_pr_auc | shot_acc | shot_tpr | shot_fpr | theta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| adv_mamba_lite_ws128_st16_e5_s42 | mamba_lite（轻量Mamba） | 0.988993 | 0.990111 | 0.913488 | 0.976879 | 0.894737 | 0.000000 | 0.767124 |
| adv_transformer_small_ws128_st16_e5_s42 | transformer_small（小型Transformer） | 0.986955 | 0.977404 | 0.879365 | 0.959538 | 0.842105 | 0.007407 | 0.826120 |
| adv_gru_ws128_st16_e5_s42 | gru（GRU门控循环单元） | 0.984509 | 0.986405 | 0.760785 | 0.924855 | 0.684211 | 0.007407 | 0.858896 |
