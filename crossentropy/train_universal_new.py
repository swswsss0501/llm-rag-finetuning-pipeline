import torch
import os
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
from datasets import load_dataset

# ==============================================================================
# 0. 导入自定义 Triton 算子
# ==============================================================================
try:
    from fused_ce_kernel import TritonCrossEntropyLoss
    print("[System] ✅ 成功加载自定义 Triton 算子 (fused_ce_kernel)。")
except ImportError as e:
    print(f"[Error] ❌ 无法导入 fused_ce_kernel。请确保文件在当前目录下。错误: {e}")
    sys.exit(1)

# ==============================================================================
# 1. 全局配置与路径
# ==============================================================================
MODEL_PATH = "/mnt/backup/models/mixtral-8x7b-instruct-v0.1"
DATA_PATH = "train_data_cot_refined.jsonl"
OUTPUT_DIR = "/mnt/backup/models/outputs_mixtral_moe_4bit_triton" 

# 显存控制参数 (针对 A100 80G + NF4)
MAX_SEQ_LENGTH = 8192       
PER_DEVICE_BATCH_SIZE = 2   
GRADIENT_ACCUMULATION = 16  
LEARNING_RATE = 2e-4        
MAX_STEPS = 60              
WARMUP_STEPS = 10

# ==============================================================================
# 2. 自定义 Trainer
# ==============================================================================
class UniversalTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused_loss_fct = TritonCrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"] 
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.fused_loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# 3. 数据格式化 (核心修复：适配嵌套结构)
# ==============================================================================
def create_formatting_func(tokenizer):
    def formatting_prompts_func(examples):
        # 【修复】这里不再读取 flat 的 instruction/output
        # 而是读取嵌套的 qa_pair 和 retrieved_context
        qa_pairs = examples["qa_pair"]           # 列表包含字典
        contexts = examples["retrieved_context"] # 列表包含字符串
        
        texts = []
        for qa, ctx in zip(qa_pairs, contexts):
            # 解析嵌套字段
            question = qa["question"]
            # 将思维链步骤合并为文本
            cot_content = "\n".join(qa["cot_steps"])
            final_ans = qa["final_answer"]
            
            # 构建 Input (RAG Context + Question)
            user_input = f"基于以下参考文档回答问题：\n{ctx}\n\n问题：{question}"
            
            # 构建 Output (CoT + Answer)
            # 这里强制模型输出“思考过程”和“最终结论”
            assistant_output = f"【思考过程】\n{cot_content}\n\n【最终结论】\n{final_ans}"
            
            # 拼装 ChatML 格式
            text = f"""<|im_start|>system
You are a helpful AI assistant specialized in military intelligence analysis.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{assistant_output}<|im_end|>""" + tokenizer.eos_token
            
            texts.append(text)
        return { "text" : texts }
    return formatting_prompts_func

# ==============================================================================
# 4. 主流程
# ==============================================================================
if __name__ == "__main__":
    
    # --- A. 优先检查路径 ---
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 数据集不存在: {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)

    # --- B. 优先加载 Tokenizer & 处理数据 (Fail-Fast 策略) ---
    print(f"\n[Data] 正在加载 Tokenizer 和 数据集 (在加载大模型之前)...")
    
    # 仅加载 Tokenizer，非常快
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token 
    
    # 加载并映射数据
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    try:
        # 这里如果报错，会立刻抛出，不会等到模型加载完
        dataset = dataset.map(create_formatting_func(tokenizer), batched=True)
        print(f"[Data] ✅ 数据处理成功！样本示例：\n{dataset[0]['text'][:200]}...")
    except Exception as e:
        print(f"\n[Error] ❌ 数据映射失败！请检查字段名。错误信息:\n{e}")
        sys.exit(1)

    # --- C. 4-bit NF4 量化配置 ---
    print(f"\n[Loader] 数据检查通过，开始加载 4-bit 模型 (A100 优化版)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                   
        bnb_4bit_quant_type="nf4",           
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    # --- D. 加载模型 (耗时操作放在最后) ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" 
    )
    
    # 开启梯度检查点
    model = prepare_model_for_kbit_training(model)

    # --- E. 配置 LoRA ---
    print(f"[Loader] 配置 LoRA 适配器 (MoE 增强版)...")
    peft_config = LoraConfig(
        r = 64,             
        lora_alpha = 128,   
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   
            "gate_proj", "up_proj", "down_proj",      
        ],
        lora_dropout = 0.05,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- F. 训练参数设置 ---
    training_args = TrainingArguments(
        per_device_train_batch_size = PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        warmup_steps = WARMUP_STEPS,
        max_steps = MAX_STEPS,
        learning_rate = LEARNING_RATE,
        fp16 = False,   
        bf16 = True,    
        logging_steps = 1,
        optim = "paged_adamw_32bit", 
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", 
        output_dir = OUTPUT_DIR,
        report_to = "none",
        gradient_checkpointing=True, 
    )

    # --- G. 启动 Trainer ---
    trainer = UniversalTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH, 
        dataset_num_proc = 2,
        packing = False, 
        args = training_args,
    )

    print(f"\n{'='*60}")
    print(f"  开始训练 (4-bit NF4 Mode + FlashAttn2 + Triton Loss)")
    print(f"{'='*60}\n")
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial VRAM Used (Weights Only): {gpu_mem:.2f} GB")

    trainer.train()

    # --- H. 保存结果 ---
    print(f"[System] 保存模型至 {OUTPUT_DIR} ...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    print(f"[System] 训练结束。")