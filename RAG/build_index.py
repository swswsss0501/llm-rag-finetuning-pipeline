import os
import glob
import json
import uuid
import torch
from typing import List, Dict

# LangChain Imports - 已更新以消除警告
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# [修正] 使用新的库导入 HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ================= 配置区域 =================
# [注意] 请确保这里指向你真实存在的 JSON 文件路径
JSON_PATH = "all_cot_results_final.json"  
DOC_DIR = "knowledge_base"
INDEX_SAVE_PATH = "rag_index_faiss"

# 这是一个轻量级的 Embedding 模型 (约 500MB - 1GB)，用于生成向量索引
EMBEDDING_MODEL = "BAAI/bge-m3" 

# ================= 辅助函数 =================

def normalize_filename(path):
    """从路径中提取纯文件名（不含扩展名），用于模糊匹配"""
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def find_parent_chunk(quote: str, chunks: List[Document], threshold=0.6):
    """
    在原始切片中寻找包含 quote 的片段。
    """
    if not quote:
        return None
    
    # 策略1: 直接包含匹配
    for chunk in chunks:
        if quote in chunk.page_content:
            return chunk.page_content
    
    # 策略2: 尝试匹配前 20 个字符
    short_quote = quote[:20]
    for chunk in chunks:
        if short_quote in chunk.page_content:
            return chunk.page_content
            
    return None

# ================= 主程序 =================

def main():
    print(f"Checking directory: {os.path.abspath(DOC_DIR)}")
    print(f"Checking JSON: {os.path.abspath(JSON_PATH)}")

    # 1. 检查目录和 JSON
    if not os.path.exists(DOC_DIR):
        print(f"[错误] 文档目录 {DOC_DIR} 不存在！")
        return
    
    json_data = []
    if not os.path.exists(JSON_PATH):
        print(f"[错误] JSON 文件 {JSON_PATH} 依然未找到！请检查路径。")
        # 这里我们选择报错退出，因为没有 JSON 就失去了高精索引的意义
        return
    else:
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if "type: uploaded file" in content:
                    start_idx = content.find('[')
                    content = content[start_idx:]
                json_data = json.loads(content)
            print(f"   -> ✅ 成功加载结构化情报: {len(json_data)} 条记录")
        except Exception as e:
            print(f"[错误] JSON 解析失败: {e}")
            return

    # 2. 加载原始文档 (The Parent Layer)
    print("\n>>> [1/5] 正在扫描并加载原始文档 (Raw Layer)...")
    raw_documents = []
    
    loaders = {
        "*.md": TextLoader,
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader
    }
    
    for pattern, LoaderClass in loaders.items():
        files = glob.glob(os.path.join(DOC_DIR, pattern))
        if files:
            print(f"   -> 发现 {len(files)} 个 {pattern} 文件")
            for file_path in files:
                try:
                    if LoaderClass == TextLoader:
                        loader = LoaderClass(file_path, encoding='utf-8')
                    else:
                        loader = LoaderClass(file_path)
                    raw_documents.extend(loader.load())
                except Exception as e:
                    print(f"   [警告] 无法读取 {file_path}: {e}")

    if not raw_documents:
        print("[错误] 未加载到任何文档，终止。")
        return

    # 3. 切分原始文档
    print(f"\n>>> [2/5] 正在切分原始文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    raw_chunks = text_splitter.split_documents(raw_documents)
    print(f"   -> 原始文档被切分为 {len(raw_chunks)} 个片段")

    # 建立映射
    file_chunk_map = {}
    for chunk in raw_chunks:
        chunk.metadata["doc_type"] = "raw_fallback"
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        fname = normalize_filename(chunk.metadata.get("source", ""))
        if fname not in file_chunk_map:
            file_chunk_map[fname] = []
        file_chunk_map[fname].append(chunk)

    # 4. 构建高精子索引
    print(f"\n>>> [3/5] 正在构建高精子索引 (Processing JSON Pointers)...")
    pointer_docs = []
    
    for entry in json_data:
        json_fname = normalize_filename(entry.get("file_name", ""))
        target_chunks = []
        for raw_fname, chunks in file_chunk_map.items():
            if json_fname in raw_fname or raw_fname in json_fname:
                target_chunks = chunks
                break
        
        # --- Spec Index ---
        if "refined_details" in entry:
            for item in entry["refined_details"]:
                vector_text = f"{item.get('subject', '')} {item.get('attribute_name', '')} {item.get('value', '')}"
                quote = item.get("quote", {}).get("text", "")
                parent_context = find_parent_chunk(quote, target_chunks)
                final_context = parent_context if parent_context else item.get("logic_chain", vector_text)

                doc = Document(
                    page_content=vector_text,
                    metadata={
                        "doc_type": "spec_pointer",
                        "entity": item.get('subject'),
                        "category": item.get('attribute_category'),
                        "context": final_context, # 关键字段
                        "source": entry.get("file_name")
                    }
                )
                pointer_docs.append(doc)

        # --- Logic Index ---
        if "raw_logic" in entry:
            for item in entry["raw_logic"]:
                vector_text = item.get("logic_content", "")
                anchors = item.get("evidence_anchors", [])
                parent_context = None
                if anchors:
                    parent_context = find_parent_chunk(anchors[0], target_chunks)
                final_context = parent_context if parent_context else vector_text

                doc = Document(
                    page_content=vector_text,
                    metadata={
                        "doc_type": "logic_pointer",
                        "logic_subject": item.get("logic_subject"),
                        "relation": item.get("relation_type"),
                        "context": final_context, # 关键字段
                        "source": entry.get("file_name")
                    }
                )
                pointer_docs.append(doc)

    print(f"   -> 生成了 {len(pointer_docs)} 个高精指针索引")
    
    final_docs = pointer_docs + raw_chunks
    
    # 5. 加载 Embedding 模型
    print(f"\n>>> [4/5] 准备加载 Embedding 模型: {EMBEDDING_MODEL}")
    print("   [提示] 这是一个轻量级模型 (约1GB显存)，用于将文本转换为向量，不是生成式大模型。")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   -> 使用设备: {device}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
    except Exception as e:
        print(f"[Fatal] 模型加载失败: {e}")
        return

    # 6. 生成并保存
    print(f"\n>>> [5/5] 正在计算 {len(final_docs)} 条数据的向量并建立索引...")
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    vectorstore.save_local(INDEX_SAVE_PATH)
    
    print(f"\n✅ RAG 索引构建完成！索引已保存至: ./{INDEX_SAVE_PATH}")

if __name__ == "__main__":
    main()