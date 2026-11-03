# sweep_attn_launch.py 再生于 20261103_232448（重调检查点：P2′/P3′/冻结前终扫后重跑）
LAUNCH_TABLE = {
    ("decode", "le16", "le8k"): (1, 1),
    ("decode", "le32", "le32k"): (1, 1),
    ("decode", "le32", "le8k"): (2, 1),
    ("decode", "le64", "le32k"): (1, 1),
    ("decode", "le64", "le8k"): (1, 1),
    ("prefill", "le16", "le32k"): (2, 1),
    ("prefill", "le16", "le8k"): (2, 1),
}
