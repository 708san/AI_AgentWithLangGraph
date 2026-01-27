# agent/utils/profiler.py
import time
import functools
from typing import Dict, List
from collections import defaultdict

class NodeProfiler:
    """ノードの実行時間を計測するプロファイラー"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
    
    def start(self, node_name: str):
        """ノードの実行開始"""
        self.start_times[node_name] = time.time()
    
    def end(self, node_name: str):
        """ノードの実行終了"""
        if node_name in self.start_times:
            elapsed = time.time() - self.start_times[node_name]
            self.timings[node_name].append(elapsed)
            del self.start_times[node_name]
            return elapsed
        return None
    
    def get_summary(self) -> str:
        """実行時間のサマリーを取得"""
        if not self.timings:
            return "No profiling data available."
        
        lines = ["\n" + "="*60]
        lines.append("ノード実行時間プロファイル")
        lines.append("="*60)
        
        # 合計時間でソート
        sorted_nodes = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        total_time = sum(sum(times) for times in self.timings.values())
        
        for node_name, times in sorted_nodes:
            count = len(times)
            total = sum(times)
            avg = total / count
            percentage = (total / total_time) * 100
            
            lines.append(f"\n{node_name}:")
            lines.append(f"  実行回数: {count}")
            lines.append(f"  合計時間: {total:.2f}秒 ({percentage:.1f}%)")
            lines.append(f"  平均時間: {avg:.2f}秒")
            if count > 1:
                lines.append(f"  最小/最大: {min(times):.2f}秒 / {max(times):.2f}秒")
        
        lines.append(f"\n{'='*60}")
        lines.append(f"総実行時間: {total_time:.2f}秒")
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)
    
    def reset(self):
        """プロファイルデータをリセット"""
        self.timings.clear()
        self.start_times.clear()

# グローバルインスタンス
profiler = NodeProfiler()

def profile_node(func):
    """ノード実行時間を計測するデコレーター"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        node_name = func.__name__
        profiler.start(node_name)
        try:
            result = func(*args, **kwargs)
            elapsed = profiler.end(node_name)
            print(f"[Profile] {node_name}: {elapsed:.2f}秒")
            return result
        except Exception as e:
            profiler.end(node_name)
            raise e
    return wrapper