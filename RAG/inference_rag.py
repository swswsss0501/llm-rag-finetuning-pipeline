import os
import sys
import torch
import warnings
from typing import List
from threading import Thread

# [恢复] 命令行输入增强库
# 如果没有安装，请 pip install gnureadline
try:
    import readline
    import gnureadline
except ImportError:
    pass # 如果系统不支持 gnureadline (如 Windows)，则跳过

# 屏蔽警告
warnings.filterwarnings("ignore")

# Transformers & PEFT
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel

# LangChain 相关
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
# [恢复] 专业的文本切分库 (用于智能截断上下文，而非暴力切片)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 1. 配置区域 =================
BASE_MODEL_PATH = "/mnt/backup/models/mixtral-8x7b-instruct-v0.1"
ADAPTER_PATH = "/mnt/backup/models/outputs_mixtral_moe_4bit_triton/lora_model"
INDEX_PATH = "./rag_index_faiss"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.1
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ================= 2. 核心组件：关键词停止器 =================
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list, tokenizer, input_ids_len:int):
        self.tokenizer = tokenizer
        self.start_len = input_ids_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_context = input_ids[0][self.start_len:]
        current_text = self.tokenizer.decode(current_context, skip_special_tokens=False)
        # 只要检测到输出里包含这个结束符，立刻停止
        if "<|im_end|>" in current_text:
            return True
        return False

# ================= 3. RAG 引擎类 =================

class IntelligenceRAG:
    def __init__(self):
        self._init_splitter() # 初始化分词器
        self._check_files()
        self._load_model()
        self._load_retriever()

    def _init_splitter(self):
        # [恢复] 使用 RecursiveCharacterTextSplitter 替代原来的 [:500]
        # 这样截断时会优先在段落、句子结束处切分，保留语义完整性
        self.smart_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
    def _check_files(self):
        print(">>> [1/4] 文件自检...")
        if not os.path.exists(BASE_MODEL_PATH):
            sys.exit("❌ 底座模型缺失")
        print("   ✅ 检查通过")

    def _load_model(self):
        print(">>> [2/4] 加载模型 (Reverted Tokenizer)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # 1. 正常加载 Tokenizer (不添加 special tokens，防止截断问题)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token # [修复] 解决 Attention Mask 警告

        # 2. 加载底座
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. 挂载 LoRA
        print(f"   -> 挂载 Adapter: {ADAPTER_PATH}")
        self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
        self.model.eval()
        print("   ✅ 模型加载完成")

    def _load_retriever(self):
        print(f">>> [3/4] 加载索引 ({EMBEDDING_MODEL_NAME})...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device}
            )
            self.vectorstore = FAISS.load_local(
                INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            print("   ✅ 索引加载完成")
        except Exception as e:
            sys.exit(f"❌ 索引加载失败: {e}")

    def render_context(self, docs: List[Document]) -> str:
        rendered_texts = []
        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get("doc_type", "raw_fallback")
            source = doc.metadata.get("source", "Unknown")
            content = doc.metadata.get("context", doc.page_content)
            
            if doc_type == "spec_pointer":
                entity = doc.metadata.get("entity", "-")
                attr = doc.metadata.get("category", "-")
                text = (f"[情报源 {i+1}] 来源：{source} | 实体：{entity} | 属性：{attr} | 核心事实：{content}")
            
            elif doc_type == "logic_pointer":
                subject = doc.metadata.get("logic_subject", "-")
                text = (f"[情报源 {i+1}] 来源：{source} | 战术主题：{subject} | 逻辑推演：{content}")
            
            else:
                # [修复] 使用智能分词器进行截断，而不是 content[:500]
                # 这能保证不会在句子中间硬切断，避免产生歧义
                chunks = self.smart_splitter.split_text(content)
                safe_content = chunks[0] if chunks else content[:500]
                text = (f"[情报源 {i+1}] 来源：{source} | 内容：{safe_content}...")
            
            rendered_texts.append(text)
        return "\n\n".join(rendered_texts)

    def chat(self, user_query: str):
        print("   🔍 检索中...")
        docs = self.retriever.invoke(user_query)
        context_str = self.render_context(docs)
        
        # Prompt 模板 (严格 ChatML)
        system_prompt = "You are a helpful AI assistant specialized in military intelligence analysis."
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"基于以下参考文档回答问题：\n{context_str}\n\n"
            f"问题：{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 编码 (同时获取 input_ids 和 attention_mask)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        # 初始化关键词停止器
        stop_criteria = KeywordsStoppingCriteria(keywords_ids=[], tokenizer=self.tokenizer, input_ids_len=input_len)
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=60.0, 
            skip_prompt=True, 
            skip_special_tokens=True 
        )

        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # 显式 Mask，解决回车死循环
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            stopping_criteria=stopping_criteria_list, # 关键词停止
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        print("\n🤖 分析结果:\n" + "-"*40)
        
        # 实时打印 + 字符串级熔断
        for new_text in streamer:
            # 检测停止符字符串
            if "<|im_end|>" in new_text:
                clean_text = new_text.split("<|im_end|>")[0]
                print(clean_text, end="", flush=True)
                break 
            
            # 检测下一轮幻觉
            if "<|im_start|>" in new_text:
                clean_text = new_text.split("<|im_start|>")[0]
                print(clean_text, end="", flush=True)
                break

            print(new_text, end="", flush=True)
            
        print("\n" + "-"*40)
        print("\n📚 引用:")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.metadata.get('source')} ({doc.metadata.get('doc_type','raw')})")

# ================= 4. 主程序 =================

def main():
    rag = IntelligenceRAG()
    print("\n🚀 RAG 系统就绪 (Full Logic: Tokenizer Reverted + Mask + Smart Splitter)\n")
    while True:
        try:
            q = input("\n📝 指令 (quit退出): ").strip()
        except EOFError:
            break
        if q.lower() in ['quit', 'exit']: break
        if not q: continue
        try:
            rag.chat(q)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()