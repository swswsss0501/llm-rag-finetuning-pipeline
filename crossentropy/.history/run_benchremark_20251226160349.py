import subprocess
import os
import sys

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"🚀 正在运行: {description} ({script_name})")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(script_name):
        print(f"❌ 错误: 找不到文件 {script_name}")
        return
    
    # 启动子进程，实时输出
    try:
        # 使用当前解释器运行
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True,
            env=os.environ.copy() # 继承环境变量
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行失败，返回码: {e.returncode}")

if __name__ == "__main__":
    print(">>> 开始全量对比测试 (PyTorch Native vs Triton Fused) <<<\n")
    
    # 1. 跑 PyTorch 原生
    run_script("mem_test2.py", "PyTorch 原生实现 (Native)")
    
    # 2. 跑 Triton 融合算子
    run_script("mem_test_triton.py", "Triton 融合算子 (Fused)")
    
    print(f"\n{'='*60}")
    print("✅ 对比测试结束。请向上滚动查看【显存峰值】和【耗时】的差异。")
    print(f"{'='*60}")