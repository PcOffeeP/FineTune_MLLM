"""
性能监控模块
提供系统资源监控、Token统计和性能计时功能
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer


class PerformanceMonitor:
    """性能监控器类，统一管理资源使用和性能数据"""
    
    def __init__(self):
        """初始化监控器"""
        self.logs: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}
        
    def get_system_stats(self) -> Dict[str, Any]:
        """获取当前系统资源使用情况"""
        stats = {}
        
        # CPU和内存信息
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        stats['memory_used_gb'] = memory.used / (1024**3)
        stats['memory_total_gb'] = memory.total / (1024**3)
        stats['memory_percent'] = memory.percent
        
        # GPU信息
        if torch.cuda.is_available():
            stats['gpu_available'] = True
            # 获取当前GPU显存使用
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            
            # 获取GPU详细状态
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                stats['gpu_utilization'] = gpu_util.gpu
                stats['gpu_memory_total_mb'] = gpu_info.total / (1024**2)
                stats['gpu_memory_used_mb'] = gpu_info.used / (1024**2)
                pynvml.nvmlShutdown()
            except ImportError as e:
                print(f"[DEBUG] 导入pynvml失败: {e}")
                try:
                    import nvidia.ml as pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    stats['gpu_utilization'] = gpu_util.gpu
                    stats['gpu_memory_total_mb'] = gpu_info.total / (1024**2)
                    stats['gpu_memory_used_mb'] = gpu_info.used / (1024**2)
                    pynvml.nvmlShutdown()
                except ImportError as e2:
                    print(f"[DEBUG] 导入nvidia.ml也失败: {e2}")
                    stats['gpu_utilization'] = None
                except Exception as e3:
                    stats['gpu_utilization'] = None
            except Exception as e:
                stats['gpu_utilization'] = None
        else:
            stats['gpu_available'] = False
            
        return stats
    
    def start_timer(self, event_name: str):
        """开始计时某个事件"""
        self._start_times[event_name] = time.time()
        
    def end_timer(self, event_name: str) -> float:
        """结束计时并记录耗时"""
        if event_name not in self._start_times:
            raise ValueError(f"Timer for '{event_name}' was not started")
        
        elapsed = time.time() - self._start_times[event_name]
        self.logs[f"{event_name}_time"] = elapsed
        return elapsed
    
    def record_value(self, key: str, value: Any):
        """记录任意值到日志"""
        self.logs[key] = value
        
    def count_tokens(self, tokenizer: PreTrainedTokenizer, messages: list) -> int:
        """计算消息中的Token数量（仅文本部分）"""
        text_content = []
        for msg in messages:
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        text_content.append(item)
        
        full_text = " ".join(text_content)
        tokens = tokenizer.encode(full_text)
        token_count = len(tokens)
        
        self.logs['input_tokens'] = token_count
        return token_count
    
    def record_inference(self, tokenizer: PreTrainedTokenizer, answer: str, 
                        event_name: str = 'inference'):
        """记录推理相关的Token统计"""
        # 输出Token
        output_tokens = len(tokenizer.encode(answer))
        self.logs['output_tokens'] = output_tokens
        
        # 总Token
        input_tokens = self.logs.get('input_tokens', 0)
        self.logs['total_tokens'] = input_tokens + output_tokens
        
        # 生成速度
        inference_time = self.logs.get(f'{event_name}_time', 0)
        if inference_time > 0:
            tokens_per_sec = output_tokens / inference_time
            self.logs['tokens_per_sec'] = tokens_per_sec
    
    def print_report(self, model_path: str, gpu_enabled: bool):
        """打印完整的性能报告"""
        print("\n" + "="*60)
        print("PERFORMANCE & RESOURCE REPORT")
        print("="*60)
        
        # 模型信息
        print(f"\n[Model Info]")
        print(f"Model Path: {model_path}")
        print(f"GPU Enabled: {gpu_enabled}")
        
        # 时间统计
        print(f"\n[Time Statistics]")
        for key, value in self.logs.items():
            if key.endswith('_time'):
                print(f"  {key.replace('_time', '').title()}: {value:.2f}s")
        
        # 总执行时间
        total_time = sum(v for k, v in self.logs.items() if k.endswith('_time'))
        print(f"  Total Execution: {total_time:.2f}s")
        
        # 系统资源
        print(f"\n[System Resources]")
        stats = self.get_system_stats()
        print(f"  CPU Usage: {stats['cpu_percent']:.1f}%")
        print(f"  Memory: {stats['memory_used_gb']:.2f}GB / {stats['memory_total_gb']:.2f}GB "
              f"({stats['memory_percent']:.1f}%)")
        
        if stats['gpu_available']:
            print(f"  GPU Memory: {stats['gpu_memory_allocated_mb']:.0f}MB / "
                  f"{stats.get('gpu_memory_total_mb', 0):.0f}MB")
            if stats['gpu_utilization'] is not None:
                print(f"  GPU Utilization: {stats['gpu_utilization']:.1f}%")
        
        # Token 统计
        if 'total_tokens' in self.logs:
            print(f"\n[Token Statistics]")
            print(f"  Input Tokens: {self.logs['input_tokens']:,}")
            print(f"  Output Tokens: {self.logs['output_tokens']:,}")
            print(f"  Total Tokens: {self.logs['total_tokens']:,}")
            if 'tokens_per_sec' in self.logs:
                print(f"  Generation Speed: {self.logs['tokens_per_sec']:.2f} tokens/sec")
        
        # 视频信息
        if 'num_frames' in self.logs:
            print(f"\n[Video Info]")
            print(f"  Frames Processed: {self.logs['num_frames']}")
            print(f"  Temporal Groups: {self.logs.get('num_temporal_groups', 'N/A')}")
        
        print("\n")