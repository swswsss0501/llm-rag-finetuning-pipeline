import os
import json
import glob
import sys
import re
from typing import List, Literal, Union, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
from openai import OpenAI

# ================= 配置区域 =================

# 1. vLLM 服务配置
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY" 
MODEL_NAME = "Qwen2.5-32B"

# 2. 文件路径
INPUT_DIR = "./"
INTERMEDIATE_DIR = "./intermediate_results"  
OUTPUT_FILE = "all_cot_results_stage1_2.json"

# 双轨 Prompt 配置
PROMPT_STAGE1_DETAIL_FILE = "prompt_stage1_detail.txt"  # 🔬 轨道A：细节切片挖掘
PROMPT_STAGE1_LOGIC_FILE = "prompt_stage1_logic.txt"    # 🔭 轨道B：全篇逻辑解构
PROMPT_STAGE2_FILE = "prompt_stage2.txt"                # 🔨 Stage 2：融合精修

# 3. 生成参数
MAX_RETRIES = 3
TEMPERATURE = 0.1 
SLICE_MAX_TOKENS = 4000 # 切片大小阈值 (Token估算)

# ===========================================

# --- Pydantic 结构定义 ---

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

# [Stage 2] 最终输出结构 (保持不变)
class QuoteInfo(BaseModel):
    text: str = Field(..., description="原文短语")
    section_anchor: str = Field(..., description="章节标题")

class DocMeta(BaseModel):
    core_concept: str = Field(..., description="文档核心主题")
    strategic_goal: str = Field(..., description="战略意图摘要")
    target_audience: str = Field(..., description="目标受众")

class Stage2Item(BaseModel):
    subject: str = Field(..., description="归一化后的实体名")
    attribute_category: Literal["HARD_SPEC", "TACTICAL_CAPABILITY", "SYSTEM_LOGIC"]
    attribute_name: str
    value: str = Field(..., description="清洗后的值")
    logic_chain: str = Field(..., description="生成的因果逻辑链")
    quote: QuoteInfo
    confidence_score: int

    @field_validator('subject')
    @classmethod
    def check_subject_semantics(cls, v):
        forbidden_roots = ["主要优势", "功能概述", "战术优势", "总结", "详细参数", "特点", "简介", "参数"]
        if v in forbidden_roots:
            raise ValueError(f"Subject '{v}' 是通用词，未完成实体透视。")
        return v

    @field_validator('confidence_score')
    @classmethod
    def check_score_range(cls, v):
        if v < 1 or v > 10:
            raise ValueError(f"Confidence score {v} 超出范围 (1-10)。")
        return v

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

def call_llm(client, messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=8192
        )
        content = completion.choices[0].message.content
        if not content:
            print("      ⚠️ [警告] 模型返回内容为空！")
            return ""
        return content
    except Exception as e:
        print(f"      ❌ API 调用异常: {e}")
        return ""

# ------------------------------------------------------------------
# ⛏️ STAGE 1 (通用): 支持 Detail 和 Logic 双轨
# ------------------------------------------------------------------
def run_stage1(client, prompt_tmpl, content, task_name, model_class) -> List[Dict]:
    """
    通用 Stage 1 执行器
    :param task_name: 用于日志和文件名的标识 (e.g., "slice0_detail", "global_logic")
    :param model_class:用于校验的 Pydantic 类 (Stage1DetailItem 或 Stage1LogicItem)
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
            print(f"      💾 [DEBUG] 原始响应已保存至: {raw_save_path}")
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
                
                # 实时保存结果 (关键修改)
                parsed_save_path = os.path.join(INTERMEDIATE_DIR, f"{task_name}_parsed.json")
                with open(parsed_save_path, "w", encoding="utf-8") as f:
                    json.dump(valid_items, f, ensure_ascii=False, indent=2)
                
                return valid_items
            else:
                print(f"      ⚠️ {task_name} 解析成功但无有效数据 (Attempt {attempt+1})")
                
        except Exception as e:
             print(f"      🔄 Error (Attempt {attempt+1}): {e}")
             if json_str:
                 print(f"      🐛 Snippet: {json_str[:50]!r}...")
            
    return []

# ------------------------------------------------------------------
# 🔨 STAGE 2: 精修 (Refining)
# ------------------------------------------------------------------
def run_stage2(client, prompt_tmpl, merged_data: List[Dict], filename) -> Optional[Dict]:
    print(f"   🔨 [Stage 2] 正在融合精修与生成 CoT (输入条目数: {len(merged_data)})...")
    
    stage1_json_str = json.dumps(merged_data, ensure_ascii=False, indent=2)
    user_prompt = prompt_tmpl.replace("{{CONTENT}}", stage1_json_str)
    
    messages = [{"role": "system", "content": "你是一个高级情报架构师。"},
                {"role": "user", "content": user_prompt}]
    
    final_committed_items = []
    current_doc_meta = None
    processed_keys = set()

    for attempt in range(MAX_RETRIES):
        raw_response = call_llm(client, messages)
        json_str = extract_json_block(raw_response)
        
        try:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # ----------------- 修改开始 -----------------
                # 1. 打印失败的 JSON 片段，方便调试
                print(f"      🐛 [Debug] JSON 解析失败。失败片段(前500字符): {json_str[:500]!r}...")
                
                # 2. 抛出异常
                # 这会被外层的 except Exception 捕获，从而触发下一次循环 (Attempt + 1)
                raise ValueError(f"JSON 解析失败 (可能截断): {e}")
                # ----------------- 修改结束 -----------------
            
            # 1. 提取 Meta
            if not current_doc_meta and isinstance(data, dict) and "doc_meta" in data:
                try:
                    current_doc_meta = DocMeta(**data["doc_meta"]).model_dump()
                except ValidationError:
                    pass 
            
            # 2. 提取 Items
            items_data = []
            if isinstance(data, dict):
                items_data = data.get("items", [])
            elif isinstance(data, list):
                items_data = data
            
            if not items_data and not current_doc_meta:
                 raise ValueError("未提取到任何有效结构")

            # 3. 校验
            current_valid_items = []
            current_errors = []

            for item in items_data:
                subj = item.get('subject', 'unknown')
                attr = item.get('attribute_name', 'unknown')
                unique_key = f"{subj}_{attr}"
                
                if unique_key in processed_keys:
                    continue

                try:
                    valid_item = Stage2Item(**item)
                    current_valid_items.append(valid_item)
                    processed_keys.add(unique_key)
                except ValidationError as e:
                    err_msg = str(e).splitlines()[0]
                    current_errors.append((item, err_msg))
            
            if current_valid_items:
                final_committed_items.extend([i.model_dump() for i in current_valid_items])
                print(f"      ✅ Stage 2 新增 {len(current_valid_items)} 条高保真情报 (累计: {len(final_committed_items)})")

            # 4. 决策
            if not current_errors:
                if not current_doc_meta: 
                     messages.append({"role": "user", "content": "Items 已处理完成，但缺少 'doc_meta'。请补充文档元数据。"})
                     continue
                break 
            else:
                print(f"      ⚠️ 发现 {len(current_errors)} 条不合规，请求修补...")
                error_report = "\n".join([f"- Err: {msg}" for bad, msg in current_errors[:3]])
                messages.append({"role": "assistant", "content": raw_response})
                repair_prompt = (
                    f"上一轮有 {len(current_errors)} 条数据错误。\n"
                    f"错误示例: {error_report}\n"
                    f"请修正 JSON 格式，仅输出修正后的 items。"
                )
                messages.append({"role": "user", "content": repair_prompt})
                
        except Exception as e:
            print(f"      🔄 Stage 2 Error: {e}")
            if attempt < MAX_RETRIES - 1:
                messages.append({"role": "user", "content": "JSON 格式解析失败，请重试并输出完整的 JSON Object。"})

    if not final_committed_items and not current_doc_meta:
        return None
        
    return {
        "doc_meta": current_doc_meta if current_doc_meta else {},
        "items": final_committed_items
    }

# ------------------------------------------------------------------
# 🚀 主流程
# ------------------------------------------------------------------
def main():
    ensure_dir(INTERMEDIATE_DIR)
    
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    
    # 加载三个 Prompt
    prompt_detail = load_text(PROMPT_STAGE1_DETAIL_FILE)
    prompt_logic = load_text(PROMPT_STAGE1_LOGIC_FILE)
    prompt_refine = load_text(PROMPT_STAGE2_FILE)
    
    # 初始化切片器
    splitter = MarkdownSplitter(max_tokens=SLICE_MAX_TOKENS)
    
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    if not md_files:
        print(f"❌ 未找到 .md 文件")
        return

    all_final_results = []

    for index, file_path in enumerate(md_files):
        filename = os.path.basename(file_path)
        print(f"\n[{index+1}/{len(md_files)}] 🚀 启动双轨分析流水线: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_doc_content = f.read()
            
        stage1_merged_items = []

        # === 轨道 A: 细节切片挖掘 (Track A - Detail) ===
        print("   👉 [Track A] 进入微观细节挖掘轨道...")
        slices = splitter.process(raw_doc_content)
        print(f"      文档被切分为 {len(slices)} 个切片。")
        
        for i, doc_slice in enumerate(slices):
            task_name = f"{filename}_slice{i}_detail"
            # 调用 Stage 1，使用 Detail Item 校验
            slice_items = run_stage1(client, prompt_detail, doc_slice, task_name, Stage1DetailItem)
            stage1_merged_items.extend(slice_items)
            
        # === 轨道 B: 宏观逻辑解构 (Track B - Logic) ===
        print("   👉 [Track B] 进入宏观逻辑解构轨道...")
        task_name_logic = f"{filename}_global_logic"
        # 调用 Stage 1，使用 Logic Item 校验，输入全文
        logic_items = run_stage1(client, prompt_logic, raw_doc_content, task_name_logic, Stage1LogicItem)
        stage1_merged_items.extend(logic_items)
        
        # === 检查点 ===
        if not stage1_merged_items:
            print(f"   🚨 双轨挖掘均未提取到数据，跳过 Stage 2。")
            continue
            
        print(f"   📊 Stage 1 完成。合计提取: {len(stage1_merged_items)} 条 (细节+逻辑)。")

        # === Stage 2: 融合精修 (Refine) ===
        stage2_output = run_stage2(client, prompt_refine, stage1_merged_items, filename)
        
        if stage2_output:
            all_final_results.append({
                "file_name": filename,
                "data": stage2_output
            })
        else:
            print(f"   🚨 Stage 2 处理失败。")

    print(f"\n💾 保存最终结果到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_final_results, f, ensure_ascii=False, indent=4)
        
    print("🎉 双轨处理完成！")

if __name__ == "__main__":
    main()