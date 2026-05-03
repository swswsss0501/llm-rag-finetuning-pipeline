import os
import json
import glob
import sys
import re
from typing import List, Literal, Union, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
from openai import OpenAI

# ================= 配置区域 =================

# 0. [新增] 获取脚本所在的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. vLLM 服务配置
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY" 
MODEL_NAME = "Qwen2.5-32B"

# 2. 文件路径
INTERMEDIATE_DIR = "./intermediate_results" 
# 输入目录：读取 Stage 1 生成的 JSON 片段
STAGE1_OUTPUT_DIR = os.path.join(INTERMEDIATE_DIR, "stage1_outputs")
# [新增] Stage 2 中间分块结果保存目录 (用于回溯和调试)
STAGE2_INTERMEDIATE_DIR = os.path.join(INTERMEDIATE_DIR, "stage2_im")
# 输出文件：最终合并的高保真结果
OUTPUT_FILE = "all_cot_results_final.json"

# Prompt 配置
# [修改] 使用绝对路径，确保在任何目录运行都不报错
PROMPT_STAGE2_SYSTEM_FILE = os.path.join(SCRIPT_DIR, "prompt_stage2_system.txt")
PROMPT_STAGE2_USER_FILE = os.path.join(SCRIPT_DIR, "prompt_stage2_user.txt")

# 3. 生成参数
MAX_RETRIES = 3
TEMPERATURE = 0.1 

# [重要] 分块配置
STAGE2_BATCH_SIZE = 15 

# ===========================================

# --- Pydantic 结构定义 (Stage 2) ---
class DroppedItem(BaseModel):
    source_id: int = Field(..., description="对应输入的ID")
    reason: str = Field(..., description="丢弃原因（如：营销废话、重复、无法修复的模糊指代）")
    
class QuoteInfo(BaseModel):
    text: str = Field(..., description="原文短语")
    section_anchor: str = Field(..., description="章节标题")

class DocMeta(BaseModel):
    core_concept: str = Field(..., description="文档核心主题")
    strategic_goal: str = Field(..., description="战略意图摘要")
    target_audience: str = Field(..., description="目标受众")

class Stage2Item(BaseModel):
    source_id: int = Field(..., description="对应输入的ID")
    subject: str = Field(..., description="归一化后的实体名")

    # [修改点 1] 类型改为 str，不再使用 Literal 硬编码
    attribute_category: str = Field(..., description="硬指标/战术能力/系统机理")
    
    attribute_name: str
    value: str = Field(..., description="清洗后的值")
    logic_chain: str = Field(..., description="生成的因果逻辑链")
    quote: QuoteInfo
    confidence_score: int

    # [修改点 2] 新增校验器，实现中英文自动归一化
    @field_validator('attribute_category', mode='before')
    @classmethod
    def normalize_attribute_category(cls, v):
        # 定义映射表 (支持 英文大写、英文小写、中文)
        mapping = {
            "HARD_SPEC": "硬指标",
            "TACTICAL_CAPABILITY": "战术能力",
            "SYSTEM_LOGIC": "系统机理",
            # 兼容小写以防万一
            "hard_spec": "硬指标",
            "tactical_capability": "战术能力",
            "system_logic": "系统机理",
            # 中文保持原样
            "硬指标": "硬指标",
            "战术能力": "战术能力",
            "系统机理": "系统机理"
        }
        
        # 1. 尝试直接匹配
        if v in mapping:
            return mapping[v]
        
        # 2. 尝试大写匹配 (处理 MixedCase)
        v_upper = v.upper()
        if v_upper in mapping:
            return mapping[v_upper]

        # 3. 如果都不匹配，抛出详细错误
        valid_keys = list(set(mapping.values())) # ["硬指标", "战术能力", "系统机理"]
        raise ValueError(f"Category '{v}' 不合法。必须是以下之一或其英文代码: {valid_keys}")

    @field_validator('subject')
    @classmethod
    def check_subject_semantics(cls, v):
        # ... (保持原有逻辑不变) ...
        forbidden_roots = ["主要优势", "功能概述", "战术优势", "总结", "详细参数", "特点", "简介", "参数"]
        if v in forbidden_roots:
            raise ValueError(f"Subject '{v}' 是通用词，未完成实体透视。")
        return v

    @field_validator('confidence_score')
    @classmethod
    def check_score_quality(cls, v):
        # ... (保持原有逻辑不变) ...
        if v < 1 or v > 10:
            raise ValueError(f"Confidence score {v} 超出范围 (1-10)。")
        if v < 6:
            raise ValueError(f"质量评分过低 ({v}/10)，判定为无效数据(主体模糊/含营销词/无逻辑)。请深化逻辑链并去营销化后重新提交。")
        return v
# ===========================================

# --- 辅助工具类 ---

def load_text(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_json_block(text):
    if not text: 
        return ""
    pattern_full = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern_full, text)
    if match:
        return match.group(1)
    pattern_loose = r"```(?:json)?\s*([\s\S]*)"
    match = re.search(pattern_loose, text)
    if match:
        return match.group(1)
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        return text
    return text

def call_llm(client, messages, max_tokens=8192):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
        content = completion.choices[0].message.content
        if not content:
            print("      ⚠️ [警告] 模型返回内容为空！")
            return ""
        return content
    except Exception as e:
        print(f"      ❌ API 调用异常: {e}")
        return ""

def batch_data(data, batch_size):
    """辅助函数：将列表切分为小块"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def format_validation_error(e: ValidationError) -> str:
    try:
        # 尝试获取第一个错误的详细信息
        error_list = e.errors()
        if error_list:
            first = error_list[0]
            loc = ".".join(map(str, first.get('loc', [])))
            msg = first.get('msg', "")
            return f"Field '{loc}': {msg}"
    except:
        pass
    return str(e).split('\n')[0]

# ------------------------------------------------------------------
# 🧠 新增功能: 基于 Logic 生成 DocMeta
# ------------------------------------------------------------------
def generate_doc_meta_from_logic(client, logic_items: List[Dict], filename: str) -> Dict:
    """
    使用 Raw Logic 数据生成文档的宏观元数据 (DocMeta)
    """
    print(f"   🧠 [Meta生成] 正在基于 {len(logic_items)} 条逻辑节点分析文档宏观画像...")
    
    # 构造 Prompt
    # 注意：Logic Items 已经是结构化的，我们将其序列化给模型
    logic_str = json.dumps(logic_items, ensure_ascii=False, indent=2)
    
    # 截断保护：如果 Logic 太长，只取前 20 条 (通常包含了 Core Thesis 和 主要架构)
    if len(logic_items) > 20:
         logic_subset = logic_items[:20]
         logic_str = json.dumps(logic_subset, ensure_ascii=False, indent=2)
         print(f"      ⚠️ Logic 条目较多，已截取前 20 条用于 Meta 生成。")

    system_prompt = "你是一名战略情报分析师。请根据提供的文档逻辑拓扑（Logic Items），提炼文档的宏观元数据。"
    user_prompt = f"""
    以下是从文档中提取的宏观逻辑节点（Logic Items）：
    
    {logic_str}
    
    请分析上述逻辑，总结出该文档的元数据。
    
    请严格按照以下 JSON 格式输出（不要输出其他废话）：
    {{
        "core_concept": "文档讨论的核心技术或系统名称",
        "strategic_goal": "该技术/系统的主要战略战术意图",
        "target_audience": "该文档面向的读者群体（如：技术人员、指挥官、情报官）"
    }}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(MAX_RETRIES):
        raw_response = call_llm(client, messages, max_tokens=1024)
        json_str = extract_json_block(raw_response)
        
        try:
            data = json.loads(json_str)
            # 校验
            meta_obj = DocMeta(**data)
            print(f"      ✅ Meta 生成成功: {meta_obj.core_concept}")
            return meta_obj.model_dump()
        except Exception as e:
            print(f"      🔄 Meta 生成重试 ({attempt+1}/{MAX_RETRIES}): {e}")
    
    print(f"      ❌ Meta 生成失败，返回空对象。")
    return {}

# ------------------------------------------------------------------
# 🔨 STAGE 2: 精修 Detail (修复日志记录，增加去重追踪)
# ------------------------------------------------------------------
def run_stage2_detail_refine(client, user_prompt_tmpl, system_prompt_tmpl, merged_data: List[Dict], filename) -> List[Dict]:
    total_items = len(merged_data)
    print(f"   🔨 [Detail精修] 开始处理 {filename} (输入: {total_items} 条, Batch: {STAGE2_BATCH_SIZE})...")
    
    batches = list(batch_data(merged_data, STAGE2_BATCH_SIZE))
    final_committed_items = []
# [修改] 将 set 改为 dict，用于存储元数据以便溯源: { unique_key: { batch_idx, source_id } }
    processed_keys = {}

    # [修正缩进] 下面这一行及其后续所有内容，必须向左移动 4 个空格
    for batch_idx, batch_items in enumerate(batches):
        print(f"      👉 Batch {batch_idx + 1}/{len(batches)} ({len(batch_items)} items)...")
        
        # [步骤 A]：给当前 Batch 的数据注入临时 ID (1-based)
        batch_items_with_id = []
        for local_id, item in enumerate(batch_items, 1):
            new_item = item.copy()
            new_item['_id'] = local_id 
            batch_items_with_id.append(new_item)

        # 序列化
        stage1_json_str = json.dumps(batch_items_with_id, ensure_ascii=False, indent=2)
        
        # [修改] 使用传入的 User Prompt 模板
        user_prompt = user_prompt_tmpl.replace("{{CONTENT}}", stage1_json_str)
        
        # [修改] 直接使用传入的 System Prompt (不再硬编码)
        messages = [{"role": "system", "content": system_prompt_tmpl},
                    {"role": "user", "content": user_prompt}]

        batch_success = False
        batch_history = [] 
        batch_accumulated_valid_items = {} 
        current_errors = []

        for attempt in range(MAX_RETRIES):
            raw_response = call_llm(client, messages)
            json_str = extract_json_block(raw_response)
            
            this_attempt_valid_items = []
            this_attempt_dropped_items = []
            this_attempt_duplicates = [] # [新增] 记录被去重的数据
            current_errors = []
            
            attempt_state = {
                "attempt_index": attempt,
                "parsed_valid_items": [],
                "parsed_dropped_items": [],
                "parsed_duplicates": [], # [新增] 保存到 JSON
                "errors": [],
                "status": "pending"
            }
        
            try:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    attempt_state["status"] = "json_error"
                    raise ValueError(f"JSON 解析失败: {e}")
                
                valid_list = data.get("valid_items", [])
                dropped_list = data.get("dropped_items", [])
                
                # --- C1. 处理 Valid Items (逻辑更新) ---
                for item in valid_list:
                    if 'source_id' not in item and '_id' in item:
                        item['source_id'] = item['_id']

                    subj = item.get('subject', 'unknown')
                    attr = item.get('attribute_name', 'unknown')
                    unique_key = f"{subj}_{attr}"
                    
                # [修改] 查重逻辑：如果发现重复，提取来源信息进行打印
                    if unique_key in processed_keys:
                        prev_info = processed_keys[unique_key]
                        # 记录详细的重复来源：Batch号 + ID
                        conflict_msg = f"全局重复: 与 Batch {prev_info['batch_idx']+1} (ID {prev_info['source_id']}) 的内容冲突"
                        item['_reason'] = conflict_msg
                        this_attempt_duplicates.append(item)
                        continue

                # [新增] 防止 Batch 内部重复
                    if unique_key in batch_accumulated_valid_items:
                        # 这里比较特殊，因为 batch_accumulated_valid_items 存的是对象，我们简单报个错
                        item['_reason'] = f"Batch内重复: 当前批次内已存在 key: {unique_key}"
                        this_attempt_duplicates.append(item)
                        continue

                    try:
                        valid_item = Stage2Item(**item)
                        this_attempt_valid_items.append(valid_item)
                        batch_accumulated_valid_items[unique_key] = valid_item
                    except ValidationError as e:
                        err_msg = format_validation_error(e)
                        current_errors.append((item, err_msg))
                
                # --- C2. 处理 Dropped Items ---
                for item in dropped_list:
                    try:
                        if 'source_id' not in item and '_id' in item:
                            item['source_id'] = item['_id']
                        this_attempt_dropped_items.append(item)
                    except:
                        pass

                # 更新状态
                attempt_state["parsed_valid_items"] = [i.model_dump() for i in this_attempt_valid_items]
                attempt_state["parsed_dropped_items"] = this_attempt_dropped_items # 原始 dict 即可
                attempt_state["parsed_duplicates"] = this_attempt_duplicates # [新增]
                attempt_state["errors"] = [{"item": i, "error": msg} for i, msg in current_errors]
                

                # --- 完整性校验 (算数平衡) ---
                input_ids = set(x['_id'] for x in batch_items_with_id)
                
                # 覆盖的ID = 有效 + 丢弃 + 错误 + 重复(也是一种有效处理)
                valid_ids = set(x.source_id for x in this_attempt_valid_items)
                dropped_ids = set(d.get('source_id') for d in this_attempt_dropped_items if d.get('source_id'))
                duplicate_ids = set(d.get('source_id') for d in this_attempt_duplicates if d.get('source_id'))
                
                error_ids = set()
                for item_dict, _ in current_errors:
                    if 'source_id' in item_dict: error_ids.add(item_dict['source_id'])
                    elif '_id' in item_dict: error_ids.add(item_dict['_id'])
                
                covered_ids = valid_ids | dropped_ids | duplicate_ids | error_ids
                missing_ids = input_ids - covered_ids
                
                if missing_ids:
                    print(f"         ⚠️ [警告] 发现模型遗漏了 {len(missing_ids)} 条数据 (ID: {missing_ids})，正在申请召回...")
                    
                    # 遍历遗漏的 ID
                    for m_id in missing_ids:
                        # 1. 找回原始输入数据 (这是关键，没有原始数据模型没法重做)
                        # 我们去 batch_items_with_id 里查
                        original_input = next((item for item in batch_items_with_id if item['_id'] == m_id), None)
                        
                        if original_input:
                            # 2. 伪造一个错误记录，加入 current_errors
                            # 错误原因明确写为 "Missing..."
                            err_msg = "严重错误：你在输出中遗漏了该条数据。你必须对其进行处理（将其归类为 'valid_items' 或 'dropped_items'）。"
                            current_errors.append((original_input, err_msg))


# [打印改进] 如果有重复，打印出来源（可选）
                if not current_errors:
                    if this_attempt_dropped_items:
                        print(f"         🗑️  丢弃: {len(this_attempt_dropped_items)} 条")
                    if this_attempt_duplicates:
                        print(f"         👯  重复: {len(this_attempt_duplicates)} 条 (已跳过)")
                        # 如果你想在控制台看详情，可以解开下面注释
                        for dup in this_attempt_duplicates:
                            print(f"            - ID {dup.get('source_id')}: {dup.get('_reason')}")
                    batch_success = True
                    attempt_state["status"] = "success"
                    batch_history.append(attempt_state)
                    break 
                else:
                    attempt_state["status"] = "has_validation_errors"
                    print(f"         ⚠️ Batch {batch_idx+1} (Att {attempt}) 发现 {len(current_errors)} 条不合规，请求修补...")
                    # ... (修复 Prompt 逻辑保持不变) ...
                    error_report_lines = []
                    for bad_item, err_msg in current_errors[:3]: 
                        bad_item_str = json.dumps(bad_item, ensure_ascii=False)
                        error_report_lines.append(f"- Data: {bad_item_str}\n  Error: {err_msg}")
                    
                    error_report = "\n".join(error_report_lines)
                    messages.append({"role": "assistant", "content": raw_response})
                    repair_prompt = (
                        f"上一轮有 {len(current_errors)} 条数据被系统驳回。\n"
                        f"驳回详情(部分):\n{error_report}\n"
                        f"请针对问题进行修正。请务必保持 JSON 结构：{{ 'valid_items': [...], 'dropped_items': [...] }}，并只输出修正后的对象。"
                    )
                    messages.append({"role": "user", "content": repair_prompt})
                
            except Exception as e:
                print(f"         🔄 Error (Att {attempt}): {e}")
                attempt_state["errors"].append({"system_error": str(e)})
                if attempt < MAX_RETRIES - 1:
                    messages.append({"role": "user", "content": "JSON 格式解析失败，请重试。"})
            
            batch_history.append(attempt_state)
    
        # 入库 (Commit Phase)
        if batch_accumulated_valid_items:
            added_this_batch = 0
            for u_key, v_item in batch_accumulated_valid_items.items():
                if u_key not in processed_keys:
                    final_committed_items.append(v_item.model_dump())
                    
                    # [修改] 入库时记录“指纹”元数据，而不仅仅是Key
                    processed_keys[u_key] = {
                        "batch_idx": batch_idx,
                        "source_id": v_item.source_id,
                        "subject": v_item.subject
                    }
                    added_this_batch += 1
            print(f"      📥 入库: {added_this_batch} 条 (总计: {len(final_committed_items)})")


        #Save Log
        log_filename = f"{filename}_batch_{batch_idx}.json"
        log_path = os.path.join(STAGE2_INTERMEDIATE_DIR, log_filename)
        
        debug_log_data = {
            "batch_index": batch_idx,
            "final_status": "success" if batch_success else "failed_with_retries",
            # [新增] 保存带 ID 的原始输入数据，实现 100% 可溯源
            "input_data": batch_items_with_id, 
            "attempts_history": batch_history 
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(debug_log_data, f, ensure_ascii=False, indent=2)

    return final_committed_items

# ------------------------------------------------------------------
# 🚀 主流程 (最终重构版)
# ------------------------------------------------------------------
def main():
    # [新增] 确保 Stage 2 中间结果目录存在
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    ensure_dir(INTERMEDIATE_DIR)
    ensure_dir(STAGE2_INTERMEDIATE_DIR) 
    
    if not os.path.exists(STAGE1_OUTPUT_DIR):
        print(f"❌ 错误: 找不到 Stage 1 输出目录: {STAGE1_OUTPUT_DIR}")
        print("请先运行 batch_cot_stage1.py 生成中间数据。")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    
# [修改] 加载两个独立的 Prompt 文件
    if not os.path.exists(PROMPT_STAGE2_SYSTEM_FILE) or not os.path.exists(PROMPT_STAGE2_USER_FILE):
        print(f"❌ 错误: 找不到 Prompt 文件 (请确保 {PROMPT_STAGE2_SYSTEM_FILE} 和 {PROMPT_STAGE2_USER_FILE} 均存在)")
        return
    
    prompt_system = load_text(PROMPT_STAGE2_SYSTEM_FILE)
    prompt_user = load_text(PROMPT_STAGE2_USER_FILE)
    
# 1. 获取所有 Detail 文件 (修改：适配 Stage 1.5 指代消解后的文件名)
    detail_files = glob.glob(os.path.join(STAGE1_OUTPUT_DIR, "*_detail_stage1_5.json"))
    
    if not detail_files:
        print(f"❌ 未找到 Detail 数据文件 (*_detail_stage1_5.json)")
        return

    all_final_results = []

    for index, detail_file_path in enumerate(detail_files):
        filename_full = os.path.basename(detail_file_path)
        
        # [修改] 从文件名中剥离新的后缀，提取 base_name
        base_name = filename_full.replace("_detail_stage1_5.json", "")
        
        print(f"\n[{index+1}/{len(detail_files)}] 🚀 处理文档: {base_name}")
        
        # --- A. 准备数据源 ---
        # [修改] 逻辑文件通常未经过 Stage 1.5 处理，仍保留原名 _logic_stage1.json
        # 这里将 detail 的新后缀替换回 logic 的旧后缀以定位文件
        logic_file_path = detail_file_path.replace("_detail_stage1_5.json", "_logic_stage1.json")
        
        raw_logic_data = []
        detail_data = []
        
        # 加载 Logic
        if os.path.exists(logic_file_path):
            with open(logic_file_path, "r", encoding="utf-8") as f:
                raw_logic_data = json.load(f)
        else:
            print(f"   ⚠️ 无 Logic 文件，将跳过 Meta 生成。")

        # 加载 Detail
        with open(detail_file_path, "r", encoding="utf-8") as f:
            try:
                detail_data = json.load(f)
            except:
                print(f"   ❌ Detail 文件损坏，跳过。")
                continue

        # --- B. 生成 Doc Meta (基于 Logic) ---
        doc_meta_result = {}
        if raw_logic_data:
            # 调用专门的 Meta 生成函数
            doc_meta_result = generate_doc_meta_from_logic(client, raw_logic_data, base_name)
        
# --- C. 清洗 Details (基于 LLM Batch) ---
        refined_details_result = []
        if detail_data:
            # [修改] 传入两个 Prompt
            refined_details_result = run_stage2_detail_refine(client, prompt_user, prompt_system, detail_data, base_name)
        else:
            print(f"   ⚠️ Detail 数据为空，跳过清洗。")

        # --- D. 组装最终结果 ---
        if refined_details_result or raw_logic_data:
            final_obj = {
                "file_name": base_name,
                "doc_meta": doc_meta_result,         # 来自 Logic
                "refined_details": refined_details_result, # 来自 Detail
                "raw_logic": raw_logic_data          # 原样保留
            }
            all_final_results.append(final_obj)
        else:
            print(f"   🚨 无有效输出。")

    print(f"\n💾 保存最终结果到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_final_results, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Stage 2 完成！")

if __name__ == "__main__":
    main()