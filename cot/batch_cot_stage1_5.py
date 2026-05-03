import json
import os
import glob
from typing import List, Dict, Optional
from openai import OpenAI  # 确保已安装: pip install openai

# ================= vLLM 服务配置 =================
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen2.5-32B"  # 请确保与 vLLM 启动参数中的 --model 名称一致

# ================= 文件路径配置 =================
# 待处理 JSON 所在目录
INPUT_DIR = './intermediate_results/stage1_outputs'
# 结果保存目录
OUTPUT_DIR = './intermediate_results/stage1_outputs'
# 过程日志保存目录
LOG_DIR = './intermediate_results/stage1_5_im'
# 原始 Markdown 文档目录 (默认为当前脚本所在目录)
SOURCE_MD_DIR = '.' 
# Prompt 文件路径
PROMPT_FILE = 'prompt_stage1_5.txt'

# 初始化客户端
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
    )
except Exception as e:
    print(f"⚠️ Warning: OpenAI 客户端初始化失败: {e}")
    client = None

# ================= 核心功能函数 =================

def call_llm_api(prompt: str) -> str:
    """
    调用本地 vLLM 服务进行推理
    """
    if not client:
        return "{}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度以保证输出稳定且格式正确
            max_tokens=512,   # 限制输出长度，只需 JSON
            top_p=0.9
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"\n❌ API 调用错误: {e}")
        return "{}"

def extract_enclosing_block(md_lines: List[str], section_title: str, context_snippet: str) -> str:
    """
    全包裹式检索逻辑：
    1. 定位目标行 (Anchor)。
    2. 向上找最近的 Header (Scope Start)。
    3. 向下找最近的 Header (Scope End)。
    4. 提取中间内容。
    """
    total_lines = len(md_lines)
    target_line_idx = -1
    
    # 1. 定位锚点 (Anchor)
    # 优先匹配 section_title (去除空格比较)
    normalized_title = section_title.strip().replace(' ', '')
    for i, line in enumerate(md_lines):
        if normalized_title in line.replace(' ', ''):
            target_line_idx = i
            break
            
    # 如果 Title 没找到，尝试匹配 Context 的前20个字
    if target_line_idx == -1 and context_snippet:
        snippet_start = context_snippet.strip()[:20]
        for i, line in enumerate(md_lines):
            if snippet_start in line:
                target_line_idx = i
                break
    
    if target_line_idx == -1:
        return f"[System Warning] 无法在原文中定位章节: {section_title}"

    # 2. 向上回溯 (Backward Scan) 找父级标题
    start_idx = 0
    # 从 target 往上找，找到第一个以 # 开头的行
    for i in range(target_line_idx, -1, -1):
        line_strip = md_lines[i].strip()
        if line_strip.startswith('#'):
            start_idx = i
            break
            
    # 3. 向下延伸 (Forward Scan) 找下一个标题
    end_idx = total_lines
    # 从 target + 1 往下找，找到第一个以 # 开头的行
    for i in range(target_line_idx + 1, total_lines):
        line_strip = md_lines[i].strip()
        if line_strip.startswith('#'):
            end_idx = i
            break
            
    # 安全截断：防止 context 过长 (例如超过 300 行)
    if end_idx - start_idx > 300:
        end_idx = start_idx + 300
        
    # 提取内容
    block_lines = md_lines[start_idx:end_idx]
    return '\n'.join(block_lines)

def clean_json_response(response_text: str) -> Optional[Dict]:
    """清洗 LLM 返回的 JSON 字符串"""
    if not response_text:
        return None
        
    text = response_text.strip()
    # 去除 Markdown 代码块标记
    if text.startswith("```"):
        try:
            text = text.split('\n', 1)[1]
            if text.endswith("```"):
                text = text.rsplit('\n', 1)[0]
        except IndexError:
            pass
            
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ================= 主处理逻辑 =================

def process_single_file(json_path: str, prompt_template: str):
    file_name = os.path.basename(json_path)
    # 推导 Markdown 文件名: xx_detail_stage1.json -> xx.md
    base_name = file_name.replace('_detail_stage1.json', '')
    md_path = os.path.join(SOURCE_MD_DIR, f"{base_name}.md")
    
    # 输出路径
    output_filename = file_name.replace('_stage1.json', '_stage1_5.json')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 日志路径
    log_filename = f"{base_name}_stage1_5_process.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    if not os.path.exists(md_path):
        print(f"⚠️  跳过: 未找到对应的 MD 文件 -> {md_path}")
        return

    print(f"🚀 正在处理: {file_name}")
    print(f"   └── 源文档: {base_name}.md")
    print(f"   └── 日志: {log_path}")

    # 读取文件
    with open(md_path, 'r', encoding='utf-8') as f:
        md_lines = f.readlines() # 按行读取方便索引

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = []
    change_count = 0
    
    # 打开日志文件准备写入
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== Stage 1.5 Process Log for {file_name} ===\n")
        log_file.write(f"Model: {MODEL_NAME} @ {API_BASE}\n")
        log_file.write(f"Total Items: {len(data)}\n")
        log_file.write("="*60 + "\n\n")

        for index, item in enumerate(data):
            original_subject = item.get('raw_subject', '')
            original_attribute = item.get('attribute', '')
            
            # 1. 提取上下文块 (Enclosing Block)
            context_block = extract_enclosing_block(
                md_lines, 
                item.get('section_title', ''), 
                item.get('raw_context', '')
            )
            
            # 2. 构造 Prompt 填充项 (Target Item 仅需部分字段)
            target_item_min = {
                "raw_subject": original_subject,
                "attribute": original_attribute,
                "raw_value": item.get("raw_value", ""),
                "raw_context": item.get("raw_context", "")
            }
            target_json_str = json.dumps(target_item_min, ensure_ascii=False, indent=2)
            
            # 3. 填充 Prompt
            full_prompt = prompt_template.replace("{context_block}", context_block)
            full_prompt = full_prompt.replace("{target_item_json}", target_json_str)
            
            # === [LOG UPDATE] 记录完整输入 ===
            log_file.write(f"--- Item #{index + 1} ---\n")
            log_file.write(f"[ORIGINAL DATA]:\n")
            log_file.write(f"  Subject:   {original_subject}\n")
            log_file.write(f"  Attribute: {original_attribute}\n")
            
            log_file.write(f"\n[INPUT CONTEXT BLOCK]:\n")
            log_file.write("-" * 20 + "\n")
            log_file.write(context_block.strip()) # 记录完整的 Context 内容
            log_file.write("\n" + "-" * 20 + "\n")
            
            log_file.write(f"\n[INPUT TARGET JSON]:\n{target_json_str}\n")
            
            # 4. 调用 LLM
            llm_raw_response = call_llm_api(full_prompt)
            
            # 5. 解析结果
            fixed_item = clean_json_response(llm_raw_response)
            
            # 记录 LLM 原始输出
            log_file.write(f"\n[LLM RAW RESPONSE]:\n{llm_raw_response.strip()}\n")
            
            # 6. 更新逻辑与对比记录
            final_subject = original_subject
            final_attribute = original_attribute
            status = "NO_CHANGE"

            if fixed_item and 'raw_subject' in fixed_item:
                new_subject = fixed_item['raw_subject'].strip()
                new_attribute = fixed_item.get('attribute', original_attribute).strip()
                
                # 判断是否发生实质性变化
                subject_changed = new_subject != original_subject
                attribute_changed = new_attribute != original_attribute
                
                if subject_changed or attribute_changed:
                    item['raw_subject'] = new_subject
                    item['attribute'] = new_attribute
                    final_subject = new_subject
                    final_attribute = new_attribute
                    status = "MODIFIED"
                    change_count += 1
                else:
                    status = "KEPT_ORIGINAL"
            else:
                status = "PARSE_ERROR"
                log_file.write("[Error]: Failed to parse JSON response.\n")

            # === [LOG UPDATE] 详细对比 ===
            log_file.write(f"\n[DECISION]: {status}\n")
            if status == "MODIFIED":
                log_file.write(f"  Subject Change:   '{original_subject}' -> '{final_subject}'\n")
                log_file.write(f"  Attribute Change: '{original_attribute}' -> '{final_attribute}'\n")
            else:
                log_file.write(f"  Reason: LLM output matched original or parse failed.\n")
            
            log_file.write("\n" + "="*40 + "\n\n")
            
            updated_data.append(item)
            
            # 进度打印
            if (index + 1) % 10 == 0:
                print(f"   ...已处理 {index + 1}/{len(data)} 条")

    # 保存新的 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成 {file_name}。修改条目: {change_count}/{len(data)}。")

def main():
    # 检查 Prompt 文件是否存在
    if not os.path.exists(PROMPT_FILE):
        print(f"❌ 错误: 未在根目录下找到 Prompt 文件: {PROMPT_FILE}")
        return
        
    # 读取 Prompt 模板
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 查找输入文件
    json_files = glob.glob(os.path.join(INPUT_DIR, '*_detail_stage1.json'))
    
    if not json_files:
        print(f"❌ 在 {INPUT_DIR} 未找到 *_detail_stage1.json 文件。")
        return
        
    print(f"🔍 找到 {len(json_files)} 个任务文件...")
    
    for json_file in json_files:
        process_single_file(json_file, prompt_template)
    
    print("\n🏁 所有任务执行完毕。")

if __name__ == "__main__":
    main()