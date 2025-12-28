"""
视频处理工具模块
提供视频帧提取、时间戳映射和分组功能
"""

import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from scipy.spatial import cKDTree

# 默认配置参数
DEFAULT_MAX_NUM_FRAMES = 90  # 视频打包后的最大帧数
DEFAULT_MAX_NUM_PACKING = 3   # 视频帧最大打包数，有效范围：1-6
DEFAULT_TIME_SCALE = 0.1      # 时间尺度因子


def map_to_nearest_scale(values: List[float], scale: np.ndarray) -> np.ndarray:
    """
    将数值映射到最近的预定义尺度值
    
    Args:
        values: 输入的数值列表
        scale: 预定义的尺度数组
        
    Returns:
        映射后的尺度值数组
    """
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr: List, size: int) -> List[List]:
    """
    将数组按指定大小分组
    
    Args:
        arr: 输入数组
        size: 每组大小
        
    Returns:
        分组后的二维列表
    """
    return [arr[i:i+size] for i in range(0, len(arr), size)]


def uniform_sample(frame_indices: List[int], num_samples: int) -> List[int]:
    """
    从帧索引列表中均匀采样指定数量的帧
    
    Args:
        frame_indices: 帧索引列表
        num_samples: 采样数量
        
    Returns:
        采样后的帧索引列表
    """
    gap = len(frame_indices) / num_samples
    idxs = [int(i * gap + gap / 2) for i in range(num_samples)]
    return [frame_indices[i] for i in idxs]


def encode_video(
    video_path: str,
    choose_fps: int = 3,
    force_packing: Optional[int] = None,
    max_num_frames: int = DEFAULT_MAX_NUM_FRAMES,
    max_num_packing: int = DEFAULT_MAX_NUM_PACKING,
    time_scale: float = DEFAULT_TIME_SCALE
) -> Tuple[List[Image.Image], List[List[int]]]:
    """
    编码视频为帧和时间ID序列，支持3D打包
    
    Args:
        video_path: 视频文件路径
        choose_fps: 目标采样帧率
        force_packing: 强制打包数量（可选）
        max_num_frames: 最大帧数限制
        max_num_packing: 最大打包数限制
        time_scale: 时间尺度因子
        
    Returns:
        frames: PIL图像列表
        frame_ts_id_group: 分组后的时间ID列表
    """
    # 读取视频元数据
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
    
    # 计算采样帧数和打包数量
    if choose_fps * int(video_duration) <= max_num_frames:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(max_num_frames, video_duration))
    else:
        packing_nums = math.ceil(video_duration * choose_fps / max_num_frames)
        if packing_nums <= max_num_packing:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(max_num_frames * max_num_packing)
            packing_nums = max_num_packing

    # 应用强制打包设置
    if force_packing is not None:
        packing_nums = min(force_packing, max_num_packing)
    
    # 打印处理信息
    print(f"[Video Info] {video_path} | Duration: {video_duration:.2f}s | FPS: {fps:.2f}")
    print(f"[Processing] Sampled frames: {choose_frames} | Packing nums: {packing_nums}")
    
    # 均匀采样帧索引
    frame_indices = list(range(len(vr)))
    sampled_indices = np.array(uniform_sample(frame_indices, choose_frames))
    
    # 提取帧数据
    frames = vr.get_batch(sampled_indices).asnumpy()
    
    # 计算时间戳ID
    frame_idx_ts = sampled_indices / fps
    scale = np.arange(0, video_duration, time_scale)
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / time_scale
    frame_ts_id = frame_ts_id.astype(np.int32)
    
    assert len(frames) == len(frame_ts_id), "帧数与时间ID数量不匹配"
    
    # 转换为PIL图像并分组时间ID
    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id.tolist(), packing_nums)
    
    return frames, frame_ts_id_group