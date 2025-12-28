# main.py
"""
主流程脚本：视频理解任务入口
调用性能监控模块实现资源追踪
"""

import torch
from transformers import AutoModel, AutoTokenizer

from utls.video_process import encode_video
from utls.performance_monitor import PerformanceMonitor


def load_model(model_path: str, monitor: PerformanceMonitor):
    """加载预训练模型和分词器"""
    print(f"{'='*60}")
    print("Loading Model & Initializing Resources...")
    print(f"{'='*60}")
    
    # 记录加载前系统状态
    stats_before = monitor.get_system_stats()
    print(f"\n[Before Loading]")
    print(f"CPU: {stats_before['cpu_percent']:.1f}% | "
          f"Memory: {stats_before['memory_used_gb']:.2f}GB")
    
    monitor.start_timer('model_load')
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    
    monitor.end_timer('model_load')
    stats_after = monitor.get_system_stats()
    
    print(f"[After Loading]")
    print(f"CPU: {stats_after['cpu_percent']:.1f}% | "
          f"Memory: {stats_after['memory_used_gb']:.2f}GB | "
          f"GPU: {stats_after['gpu_memory_allocated_mb']:.0f}MB")
    print(f"{'='*60}\n")
    
    return model, tokenizer


def main():
    """主函数：执行视频理解任务"""
    # ==================== 配置参数 ====================
    MODEL_PATH = 'Model/MiniCPM-VL-4.5'
    VIDEO_PATH = "data/Video/testing_center/CFC_20251021_3.mp4"
    SAMPLE_FPS = 5
    FORCE_PACKING = None
    QUESTION = "该视频是材料拉伸测试过程，请分析试样的断裂时刻及位置"
    
    # ==================== 初始化监控器 ====================
    monitor = PerformanceMonitor()
    
    # ==================== 加载模型 ====================
    model, tokenizer = load_model(MODEL_PATH, monitor)
    
    # ==================== 处理视频 ====================
    print("Processing video...")
    monitor.start_timer('video_process')
    
    frames, frame_ts_id_group = encode_video(
        video_path=VIDEO_PATH,
        choose_fps=SAMPLE_FPS,
        force_packing=FORCE_PACKING
    )
    
    monitor.end_timer('video_process')
    monitor.record_value('num_frames', len(frames))
    monitor.record_value('num_temporal_groups', len(frame_ts_id_group))
    print(f"Video processing completed in {monitor.logs['video_process_time']:.2f}s\n")
    
    # ==================== 构建消息 ====================
    msgs = [{'role': 'user', 'content': frames + [QUESTION]}]
    
    # 统计输入Token
    monitor.count_tokens(tokenizer, msgs)
    
    # ==================== 模型推理 ====================
    print("Generating answer...")
    monitor.start_timer('inference')
    
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        use_image_id=False,
        max_slice_nums=1,
        temporal_ids=frame_ts_id_group
    )
    
    monitor.end_timer('inference')
    monitor.record_inference(tokenizer, answer)
    
    # ==================== 打印报告 ====================
    monitor.print_report(MODEL_PATH, torch.cuda.is_available())
    
    # 打印回答
    print("="*60)
    print("Answer:")
    print(answer)
    print("\n")


if __name__ == "__main__":
    main()
