import logging
import os
import sys

def setup_logger(save_dir, filename="train.log", distributed_rank=0):
    """
    配置全局 Logger
    Args:
        save_dir (str): 日志保存目录
        filename (str): 日志文件名
        distributed_rank (int): 分布式训练时的进程ID (LiteCue通常是单卡，默认为0)
    """
    logger = logging.getLogger("LiteCue")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 Handler (例如在 Notebook 中重复运行)
    if logger.hasHandlers():
        return logger

    # 只有主进程 (Rank 0) 才负责记录日志，避免多进程打印混乱
    if distributed_rank > 0:
        return logger

    # 1. 输出到控制台 (Stream)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # 格式: [时间] [级别] 消息
    formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. 输出到文件 (File)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, filename)
        fh = logging.FileHandler(log_file, mode='a') # 'a' for append
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger