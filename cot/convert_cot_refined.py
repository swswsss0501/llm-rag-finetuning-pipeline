import os
import json
import re
import glob
import difflib
from typing import List, Dict, Optional, Any
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. vLLM 服务配置
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY" 
MODEL_NAME = "Qwen2.5-32B"

# 2. 文件路径配置
WORK_DIR = "."  # 假设脚本、JSON、MD、Prompt都在当前目录
INPUT_JSON = os.path.join(WORK_DIR, "all_cot_results_final.json")
PROMPT_FILE = os.path.join(WORK_DIR, "prompt_stage3.txt")
OUTPUT_FILE = os.path.join(WORK_DIR, "train_data_cot_refined.jsonl")

# 3. 检索参数
CONTEXT_WINDOW_SIZE = 800  # 核心句前后扩展的字符数
MAX_CONTEXT_LEN = 3000     # 输给模型的最大上下文长度（防止爆显存）
FUZZY_MATCH_THRESHOLD = 0.6 # 章节标题模糊匹配的阈值

# ================= 工具类：Markdown 解析与检索 =================

class MarkdownMap:
    """Task A: 原文地图构建器"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.map = self._parse_file()
        self.headers = list(self.map.keys())

    @staticmethod
    def clean_header(header_text: str) -> str:
        """
        辅助工具：清洗章节标题中的数字编号，保留语义。
        输入: "2.3.6 5.3 数字化装备全生命周期管理"
        输出: "数字化装备全生命周期管理"
        """
        # 替换开头的数字、点、空格
        if not header_text:
            return ""
        return re.sub(r'^[\d\.\s]+', '', header_text)

    def _parse_file(self) -> Dict[str, str]:
        """
        解析 Markdown，将结构扁平化为 {Header: Content}。
        策略：遇到 # 开头视为新 Header，否则视为上一 Header 的正文。
        """
        structure = {}
        current_header = "Abstract_Or_Intro" # 默认开头，防止无Header的文档
        current_content = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                stripped = line.strip()
                # 识别 Header：以 # 开头
                if stripped.startswith("#"):
                    # 保存上一节
                    if current_content:
                        structure[current_header] = "\n".join(current_content)
                    
                    # 开启新一节
                    # 去除 # 号和首尾空格，保留原始脏数据格式（如 "1.2.4.2 2.2 核心能力..."）以供检索匹配
                    current_header = re.sub(r'^#+\s*', '', stripped)
                    current_content = []
                else:
                    current_content.append(stripped)
            
            # 保存最后一节
            if current_content:
                structure[current_header] = "\n".join(current_content)
                
        except Exception as e:
            print(f"[错误] 解析 Markdown 文件失败 {self.file_path}: {e}")
            return {}
            
        return structure

    def get_section_content(self, section_anchor: str) -> Optional[str]:
        """
        Task B helper: 根据 section_anchor 查找内容。
        支持：精确匹配 -> 包含匹配 -> Jaccard 模糊匹配
        """
        if not section_anchor:
            return None

        # 1. 精确匹配
        if section_anchor in self.map:
            return self.map[section_anchor]
        
        # 2. 包含匹配 (Map Key 包含 Anchor，或者 Anchor 包含 Map Key)
        for header in self.headers:
            if section_anchor in header or header in section_anchor:
                return self.map[header]
                
        # 3. 模糊匹配 (应对特殊字符或细微差异)
        best_match = difflib.get_close_matches(section_anchor, self.headers, n=1, cutoff=FUZZY_MATCH_THRESHOLD)
        if best_match:
            return self.map[best_match[0]]
            
        return None

    def search_quote_context(self, content: str, quote: str) -> str:
        """
        Task B helper: 在段落中定位 Quote 并扩展上下文窗口
        """
        if not content or not quote:
            return ""
            
        # 尝试定位
        idx = content.find(quote)
        
        # 如果找不到，尝试去标点去空格后的模糊定位（此处为简化版回退策略）
        if idx == -1:
            # 简单回退策略：如果找不到精确句子，返回整个 Section 的前 N 字符
            return content[:MAX_CONTEXT_LEN]
            
        # 扩展窗口
        start = max(0, idx - CONTEXT_WINDOW_SIZE)
        end = min(len(content), idx + len(quote) + CONTEXT_WINDOW_SIZE)
        
        return content[start:end]

# ================= 核心逻辑类 =================

class CoTPipeline:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        self.prompt_template = self._load_prompt()
        
    def _load_prompt(self):
        if not os.path.exists(PROMPT_FILE):
            print(f"[警告] 未找到 Prompt 文件 {PROMPT_FILE}，将使用默认模板。")
            return "Context:\n{context}\n\nTask: Analyze the following fact:\n{fact_data}\n\nCoT:"
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()

    def find_markdown_file(self, json_file_name: str) -> Optional[str]:
        """根据 JSON 中的 file_name 模糊查找本地 MD 文件"""
        # 1. 尝试直接拼接
        candidates = [
            f"{json_file_name}.md",
            f"{json_file_name}",
            f"{json_file_name}_extracted.md"
        ]
        
        # 2. 目录搜索
        files_in_dir = glob.glob(os.path.join(WORK_DIR, "*.md"))
        
        for cand in candidates:
            path = os.path.join(WORK_DIR, cand)
            if os.path.exists(path):
                return path
        
        # 3. 前缀匹配 (处理像 "3.AIRNEWS..." 这种文件名被截断的情况)
        clean_name = json_file_name.split('（')[0] # 去除括号后缀
        for f in files_in_dir:
            if clean_name in f:
                return f
                
        return None

    def generate_cot(self, context: str, item_data: dict, meta_info: dict) -> Optional[dict]:
        """
        Task D: 调用模型生成 CoT，并解析返回的 JSON。
        返回 None 表示生成失败、解析失败或上下文严重缺失。
        """
        if not context:
            # 即使上下文缺失，也尝试让模型判断（模型应返回 CONTEXT_INSUFFICIENT）
            context = "当前未提取到有效上下文 (Context extraction failed)。"
            
        # 构造 Prompt
        # item_data 包含 subject 和 attribute_name，供 Prompt 清洗标题使用
        fact_str = json.dumps(item_data, indent=2, ensure_ascii=False)
        
        try:
            full_prompt = self.prompt_template.format(
                context=context,
                fact_data=fact_str,
                doc_meta=json.dumps(meta_info, ensure_ascii=False)
            )
        except KeyError as e:
            print(f"[Prompt 错误] txt 文件中缺少占位符: {e}")
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    # System Prompt 已整合进 prompt_stage3.txt，这里使用通用占位
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3, # 降低温度以保证 JSON 格式稳定
                max_tokens=2048,
                extra_body={"stop_token_ids": [151643]} # Qwen 结束符
            )
            raw_content = response.choices[0].message.content.strip()
            
            # === JSON 清洗与解析逻辑 ===
            # 1. 去除 Markdown 代码块标记 ```json ... ```
            clean_content = re.sub(r'^```json\s*', '', raw_content, flags=re.MULTILINE)
            clean_content = re.sub(r'\s*```$', '', clean_content, flags=re.MULTILINE)
            
            # 2. 解析 JSON
            return json.loads(clean_content)

        except json.JSONDecodeError:
            # 简单的错误日志，防止刷屏
            # print(f"[JSON 错误] 模型输出非 JSON 格式，已跳过。") 
            return None
        except Exception as e:
            print(f"[API 错误] {e}")
            return None

    def process(self):
        # 1. 加载数据
        if not os.path.exists(INPUT_JSON):
            print(f"[错误] 输入文件不存在: {INPUT_JSON}")
            return

        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"成功加载 {len(data)} 个文档对象。开始处理...")
        
        # 计数器
        stats = {
            "generated": 0,
            "filtered": 0,
            "errors": 0
        }

        # 2. 打开输出文件 (Append 模式)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            
            # 使用 tqdm 显示总进度
            pbar = tqdm(data, desc="文档处理进度", unit="doc")
            
            for doc_item in pbar:
                file_name = doc_item.get('file_name', '')
                md_path = self.find_markdown_file(file_name)
                
                if not md_path:
                    # print(f"\n[跳过] 未找到对应的 Markdown 文件: {file_name}")
                    continue
                    
                md_map = MarkdownMap(md_path)
                doc_meta = doc_item.get('doc_meta', {})
                
                # === 处理 Refined Details (Type A: 细节参数) ===
                details = doc_item.get('refined_details', [])
                for detail in details:
                    anchor = detail.get('quote', {}).get('section_anchor', '')
                    quote_text = detail.get('quote', {}).get('text', '')
                    
                    section_content = md_map.get_section_content(anchor)
                    rich_context = md_map.search_quote_context(section_content, quote_text)
                    
                    # 降级处理：如果没有找到 Quote，尝试使用清洗后的标题作为上下文引导
                    if not rich_context:
                        clean_anchor = MarkdownMap.clean_header(anchor)
                        rich_context = f"Section: {clean_anchor}\n(原文内容提取缺失)"
                    
                    # 生成 CoT
                    cot_result = self.generate_cot(rich_context, detail, doc_meta)
                    
                    # 结果判定
                    if cot_result:
                        if cot_result.get("final_answer") == "CONTEXT_INSUFFICIENT":
                            stats["filtered"] += 1
                        else:
                            result = {
                                "type": "detail_cot",
                                "source_doc": file_name,
                                "input_fact": detail,
                                "retrieved_context": rich_context,
                                "qa_pair": cot_result
                            }
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush() # 立即写入磁盘
                            stats["generated"] += 1
                    else:
                        stats["errors"] += 1

                # === 处理 Raw Logic (Type B: 逻辑链路) ===
                logics = doc_item.get('raw_logic', [])
                for logic in logics:
                    involved = logic.get('involved_sections', '')
                    logic_context = ""
                    
                    if "全文" in involved or involved == "All":
                        # 策略：只给 Meta，不给全文，防止 Token 爆炸
                        logic_context = f"文档全貌上下文 (Full Context).\n核心概念: {doc_meta.get('core_concept')}\n战略目标: {doc_meta.get('strategic_goal')}"
                    else:
                        # 简单解析 "1.2 -> 1.3" 或 "2.1, 2.2"
                        parts = re.split(r'->|,| ', involved)
                        collected_texts = []
                        for p in parts:
                            p = p.strip()
                            if not p: continue
                            
                            # 1. 尝试用原始编号获取内容
                            content = md_map.get_section_content(p)
                            if content:
                                # 2. 在展示给模型时，清洗掉脏标题编号
                                clean_p = MarkdownMap.clean_header(p)
                                collected_texts.append(f"--- 章节: {clean_p} ---\n{content[:1000]}...")
                        
                        logic_context = "\n\n".join(collected_texts)

                    if not logic_context:
                         logic_context = "上下文提取失败。"

                    # 生成 CoT
                    cot_result = self.generate_cot(logic_context, logic, doc_meta)
                    
                    # 结果判定
                    if cot_result:
                        if cot_result.get("final_answer") == "CONTEXT_INSUFFICIENT":
                            stats["filtered"] += 1
                        else:
                            result = {
                                "type": "logic_cot",
                                "source_doc": file_name,
                                "input_fact": logic,
                                "retrieved_context": logic_context,
                                "qa_pair": cot_result
                            }
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                            stats["generated"] += 1
                    else:
                        stats["errors"] += 1

                # 更新进度条显示的统计信息
                pbar.set_postfix({
                    "已生成": stats["generated"], 
                    "已过滤": stats["filtered"]
                })

        print("\n" + "="*30)
        print("处理完成！")
        print(f"✅ 成功生成样本: {stats['generated']}")
        print(f"🗑️ 过滤无效样本: {stats['filtered']}")
        print(f"❌ 生成/解析失败: {stats['errors']}")
        print(f"📂 输出文件位置: {OUTPUT_FILE}")
        print("="*30)

if __name__ == "__main__":
    pipeline = CoTPipeline()
    pipeline.process()