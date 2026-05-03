import os
import json
import glob
import sys
import re
import random # <--- [修改点 1] 新增引用
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ================= 配置区域 =================

# 1. vLLM 服务配置
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY" 
MODEL_NAME = "Qwen2.5-32B"

# 2. 文件路径
INPUT_DIR = "./"
# 设置中间结果主目录
INTERMEDIATE_DIR = "./intermediate_results/stage1_im" 
# 设置 Stage 1 最终结果的保存子目录
STAGE1_OUTPUT_DIR = "./intermediate_results/stage1_outputs" 
# Prompt 配置 (仅保留 Stage 1)
PROMPT_STAGE1_DETAIL_FILE = "prompt_stage1_detail.txt"  # 🔬 轨道A：细节切片挖掘
PROMPT_STAGE1_LOGIC_FILE = "prompt_stage1_logic.txt"    # 🔭 轨道B：全篇逻辑解构

# 3. 生成参数
MAX_RETRIES = 3
TEMPERATURE = 0.1 
SLICE_MAX_TOKENS = 4000 # 切片大小阈值 (Token估算)

# ===========================================

# --- Pydantic 结构定义 (仅 Stage 1) ---

# [Track A] 细节挖掘结果结构
class Stage1DetailItem(BaseModel):
    raw_subject: str = Field(..., description="原文主语")
    attribute: str = Field(..., description="属性名称")
    raw_value: str = Field(..., description="原始值描述")
    raw_context: str = Field(..., description="原文上下文片段")
    section_title: str = Field(..., description="章节标题")

# [Track B] 逻辑解构结果结构
class Stage1LogicItem(BaseModel):
    logic_subject: str = Field(..., description="逻辑主体")
    relation_type: str = Field(..., description="关系类型: CAUSAL_CHAIN, SUPPORT_NET, etc.")
    topology_shape: Optional[str] = Field(None, description="拓扑形状")
    argument_layer: Optional[str] = Field(None, description="论证层级")
    support_hub: Optional[str] = Field(None, description="支撑网核心")
    support_spokes: Optional[List[str]] = Field(None, description="支撑网节点")
    logic_content: str = Field(..., description="逻辑叙述")
    evidence_anchors: Optional[List[str]] = Field(None, description="咬合点证据")
    strategic_weight: Optional[str] = Field(None, description="战略权重")
    involved_sections: Optional[str] = Field(None, description="涉及章节")

# ===========================================

# --- 辅助工具类 ---

class MarkdownSplitter:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens

    def _estimate_tokens(self, text):
        return len(text)

    def split_by_headers(self, text):
        """按Markdown标题切分原子块"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        header_pattern = re.compile(r'^#{1,3}\s+')
        
        for line in lines:
            if header_pattern.match(line) and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
            
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def group_chunks(self, chunks):
        """合并原子块为切片"""
        grouped_slices = []
        current_slice = ""
        
        for chunk in chunks:
            if self._estimate_tokens(current_slice) + self._estimate_tokens(chunk) < self.max_tokens:
                current_slice += "\n\n" + chunk
            else:
                if current_slice:
                    grouped_slices.append(current_slice)
                current_slice = chunk
        
        if current_slice:
            grouped_slices.append(current_slice)
        return grouped_slices

    def process(self, text):
        semantic_chunks = self.split_by_headers(text)
        final_slices = self.group_chunks(semantic_chunks)
        return final_slices

# ===========================================

def load_text(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_json_block(text):
    """
    鲁棒的 JSON 提取器
    """
    if not text: 
        return ""
    
    # 1. 尝试完美匹配
    pattern_full = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern_full, text)
    if match:
        return match.group(1)
    
    # 2. 尝试宽松匹配 (有头无尾 - 处理截断)
    pattern_loose = r"```(?:json)?\s*([\s\S]*)"
    match = re.search(pattern_loose, text)
    if match:
        return match.group(1)
    
    text = text.strip()
    # 3. 最后的挣扎
    if text.startswith("[") or text.startswith("{"):
        return text
        
    return text

def parse_truncated_json_list(json_str):
    """
    尝试解析截断的 JSON List
    """
    json_str = json_str.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass 
    
    if json_str.startswith("["):
        last_obj_end = json_str.rfind("},")
        if last_obj_end != -1:
            fixed_str = json_str[:last_obj_end+1] + "]"
            try:
                print(f"      🔧 [Auto-Fix] 监测到 JSON 截断，正在丢弃末尾残缺数据并封口...")
                return json.loads(fixed_str)
            except:
                pass
                
    raise ValueError("无法解析或修复的 JSON 格式")

# [修改] 基于 32k 上下文动态计算，并在压缩时显式打印
def call_llm(client, messages):
    # 1. 定义新的模型上下文上限 (用户已升级至 32768)
    MAX_MODEL_CONTEXT = 32768
    # 2. 定义我们期望的理想输出长度
    TARGET_OUTPUT_TOKENS = 8192
    # 3. 定义最低生存底线 (如果连这都给不到，说明输入太长了)
    MIN_OUTPUT_TOKENS = 512
    
    # --- 动态计算逻辑 ---
    
    # A. 估算输入 token 数 
    # (Qwen2.5 Tokenizer 平均约 1.5-1.8 字符/token，这里按 1.5 保守估算)
    all_content = "".join([str(m.get("content", "")) for m in messages])
    estimated_input_tokens = int(len(all_content) / 1.5) 
    
    # B. 计算剩余可用空间 
    # (保留 500 token 作为 Prompt 模板开销和计算误差的安全缓冲)
    available_tokens = MAX_MODEL_CONTEXT - estimated_input_tokens - 500
    
    # C. 决策 max_tokens
    if available_tokens >= TARGET_OUTPUT_TOKENS:
        # 空间充足，给足 8192
        dynamic_max_tokens = TARGET_OUTPUT_TOKENS
    else:
        # 空间不足，被迫压缩
        # 确保至少有 MIN_OUTPUT_TOKENS，否则可能会报错或截断太狠
        dynamic_max_tokens = max(MIN_OUTPUT_TOKENS, available_tokens)
        
        # [关键需求] 打印压缩日志
        print(f"      ⚠️ [动态流控] 输入预估 {estimated_input_tokens} tokens。")
        print(f"      ⚠️ [Output压缩] 剩余空间不足，max_tokens 已从 {TARGET_OUTPUT_TOKENS} 降低至 {dynamic_max_tokens}。")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=dynamic_max_tokens 
        )
        content = completion.choices[0].message.content
        if not content:
            print("      ⚠️ [警告] 模型返回内容为空！")
            return ""
        return content
    
    except Exception as e:
        print(f"      ❌ API 调用异常: {e}")
        # 如果依然出现 Context 溢出，打印更详细的调试建议
        if "max_tokens" in str(e) or "context length" in str(e):
             print(f"      💡 建议：请检查 vLLM 启动参数是否确实已添加 --max-model-len 32768")
        return ""

# ------------------------------------------------------------------
# ⛏️ STAGE 1 (通用): 支持 Detail 和 Logic 双轨
# ------------------------------------------------------------------
def run_stage1(client, prompt_tmpl, content, task_name, model_class) -> List[Dict]:
    """
    通用 Stage 1 执行器
    """
    print(f"   ⛏️ [Stage 1] 正在执行任务: {task_name} ...")
    user_prompt = prompt_tmpl.replace("{{CONTENT}}", content)
    messages = [{"role": "system", "content": "你是一个高召回率的数据挖掘专家。"},
                {"role": "user", "content": user_prompt}]
    
    for attempt in range(MAX_RETRIES):
        raw_response = call_llm(client, messages)

        # DEBUG: 保存原始响应
        if raw_response:
            raw_save_path = os.path.join(INTERMEDIATE_DIR, f"{task_name}_attempt{attempt}_raw.txt")
            with open(raw_save_path, "w", encoding="utf-8") as f:
                f.write(raw_response)
            print(f"      💾 [DEBUG] 原始响应已保存至: {raw_save_path}") # 减少刷屏
        # -----------------------------------------------------------
        json_str = extract_json_block(raw_response)
        
        try:
            if not json_str:
                if attempt < MAX_RETRIES - 1:
                    continue
                raise ValueError("未找到 JSON 内容")

            data = parse_truncated_json_list(json_str)
            
            if not isinstance(data, list):
                raise ValueError("Stage 1 必须返回 List 结构")
            
            # 动态校验
            valid_items = []
            for item in data:
                try:
                    obj = model_class(**item)
                    valid_items.append(obj.model_dump())
                except ValidationError:
                    continue 
            
            if valid_items:
                print(f"      ✅ {task_name} 捕获 {len(valid_items)} 条有效数据。")
                # 这里不需要保存 parsed_json，因为最后会统合保存
                return valid_items
            else:
                print(f"      ⚠️ {task_name} 解析成功但无有效数据 (Attempt {attempt+1})")
                
        except Exception as e:
             print(f"      🔄 Error (Attempt {attempt+1}): {e}")
            
    return []

# ------------------------------------------------------------------
# 🚀 主流程 (修改版)
# ------------------------------------------------------------------
def main():
    ensure_dir(INTERMEDIATE_DIR)
    ensure_dir(STAGE1_OUTPUT_DIR) # 确保 Stage 1 输出目录存在
    
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    
    # 加载 Stage 1 Prompts
    prompt_detail = load_text(PROMPT_STAGE1_DETAIL_FILE)
    prompt_logic = load_text(PROMPT_STAGE1_LOGIC_FILE)
    
    # 初始化切片器
    splitter = MarkdownSplitter(max_tokens=SLICE_MAX_TOKENS)
    
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    if not md_files:
        print(f"❌ 未找到 .md 文件")
        return

    for index, file_path in enumerate(md_files):
        filename = os.path.basename(file_path) # e.g. "doc1.md"
        base_name = os.path.splitext(filename)[0] # e.g. "doc1"
        
        print(f"\n[{index+1}/{len(md_files)}] 🚀 启动 Stage 1 分析: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_doc_content = f.read()
            
        # [修改点 A] 数据容器物理隔离
        all_detail_items = [] # 仅存放 Track A 细节
        all_logic_items = []  # 仅存放 Track B 逻辑

        # === 轨道 A: 细节切片挖掘 (Track A - Detail) ===
        print("   👉 [Track A] 进入微观细节挖掘轨道...")
        slices = splitter.process(raw_doc_content)
        print(f"      文档被切分为 {len(slices)} 个切片。")
        
        for i, doc_slice in enumerate(slices):
            task_name = f"{filename}_slice{i}_detail"
            # 调用 Stage 1，使用 Detail Item 校验
            slice_items = run_stage1(client, prompt_detail, doc_slice, task_name, Stage1DetailItem)
            # [修改点 B] 存入 Detail 列表
            all_detail_items.extend(slice_items)
            
        # === 轨道 B: 宏观逻辑解构 (Track B - Logic) ===
        print("   👉 [Track B] 进入宏观逻辑解构轨道...")
        task_name_logic = f"{filename}_global_logic"
        # 调用 Stage 1，使用 Logic Item 校验，输入全文
        logic_items = run_stage1(client, prompt_logic, raw_doc_content, task_name_logic, Stage1LogicItem)
        # [修改点 C] 存入 Logic 列表
        all_logic_items.extend(logic_items)
        
        # === 数据后处理与分流保存 (Refactored) ===
        
        # 1. 处理 Detail 数据 (需要 Shuffle，供 Stage 2 训练用)
        if all_detail_items:
            random.shuffle(all_detail_items) # 仅打乱 Detail
            detail_output_name = f"{base_name}_detail_stage1.json"
            detail_output_path = os.path.join(STAGE1_OUTPUT_DIR, detail_output_name)
            
            with open(detail_output_path, "w", encoding="utf-8") as f:
                json.dump(all_detail_items, f, ensure_ascii=False, indent=2)
            print(f"   💾 [保存 Detail] {len(all_detail_items)} 条 (已乱序) -> {detail_output_name}")
        else:
            print(f"   ⚠️ [警告] Track A (Detail) 未产出数据。")

        # 2. 处理 Logic 数据 (保持原序，作为原矿归档)
        if all_logic_items:
            # Logic 不打乱，保留拓扑顺序
            logic_output_name = f"{base_name}_logic_stage1.json"
            logic_output_path = os.path.join(STAGE1_OUTPUT_DIR, logic_output_name)
            
            with open(logic_output_path, "w", encoding="utf-8") as f:
                json.dump(all_logic_items, f, ensure_ascii=False, indent=2)
            print(f"   💾 [保存 Logic ] {len(all_logic_items)} 条 (保持原序) -> {logic_output_name}")
        else:
            print(f"   ⚠️ [警告] Track B (Logic) 未产出数据。")

    print("\n🎉 Stage 1 批量处理完成！")
    print("\n🎉 Stage 1 批量处理完成！请运行 batch_cot_stage2.py 进行精修。")

if __name__ == "__main__":
    main()