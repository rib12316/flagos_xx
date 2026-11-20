# Copyright (c) 2026 BAAI. All rights reserved.
"""vllm_fl 自定义 inductor pass 集（compile pass 注入通道）。

F-a rope+cat 融合见 rope_cat_fusion.py；注入点 vllm_fl/platform.py
（compilation_config.inductor_compile_config["post_grad_custom_post_pass"]），
env 门 VLLM_FL_FA_ROPECAT。
"""
