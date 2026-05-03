import subprocess
import os
import sys

def run_script(script_path, description):
    print(f"\n{'='*60}")
    print(f"🚀 正在运行: {description}")
    print(f"📂 文件路径: {script_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"❌ 错误: 找不到文件 {script_path}")
        return
    
    try:
        # 使用当前的 Python 解释器启动子进程
        # cwd=os.path.dirname(script_path) 确保子进程的工作目录在 tests/ 文件夹内
        # 这样即使你在根目录运行这个脚本，子进程也能正确找到环境
        subprocess.run(
            [sys.executable, script_path], 
            check=True,
            cwd=os.path.dirname(script_path), 
            env=os.environ.copy()
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行失败，返回码: {e.returncode}")

if __name__ == "__main__":
    print(">>> 开始全量对比测试 (PyTorch Native vs Triton Fused) <<<\n")
    
    # 1. 获取当前脚本所在目录 (即 tests/ 目录)
    # 无论你在哪里运行 python，__file__ 都能定位到这个文件的绝对位置
    current_test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 定位兄弟文件
    script_native = os.path.join(current_test_dir, "mem_test2.py")
    script_triton = os.path.join(current_test_dir, "mem_test_triton.py")
    
    # 3. 运行
    run_script(script_native, "PyTorch 原生实现 (Native)")
    run_script(script_triton, "Triton 融合算子 (Fused)")
    
    print(f"\n{'='*60}")
    print("✅ 对比测试结束。")