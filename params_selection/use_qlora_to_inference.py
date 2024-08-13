import os
import gc
import torch
import warnings
from datetime import datetime
from datasets import load_dataset

from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, TrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, PeftConfig, PeftModel # PEFT = Parameter-Efficient Fine-Tuning
from trl import SFTTrainer # trl = Transformer Reinforcement Learning

# model_name = "/home/wirl/ytc/chinese-alpaca-2-7b_自己下載的"
model_name = "/home/wirl/ytc/Taiwan-LLM-7B-v2.0.1-chat_local"

# adapter_model_path = "/home/wirl/ytc/要寫的論文研究/code/qlora/output/checkpoint-250"
# merged_model_path = "/home/wirl/ytc/要寫的論文研究/code/results/chinese-alpaca-2-7b_merged_model_by_qlora/"

# base_path = "/home/wirl/ytc/要寫的論文研究/code/qlora/output/"
base_path = "/home/wirl/ytc/要寫的論文研究/code/qlora2/output/"

# adapter_models_path = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("checkpoint-")]

# adapter_model_path = f"/home/wirl/ytc/要寫的論文研究/code/qlora/output/checkpoint-1250"
adapter_model_path = f"/home/wirl/ytc/要寫的論文研究/code/qlora2/output/checkpoint-1250"

tokenizer = LlamaTokenizer.from_pretrained(model_name, local_files_only=True, legacy=True)

def get_response(model, prompt_template, sentence_text, remove_input=True):
    device = "cuda:0"
    full_prompt = prompt_template.format(sentence_text)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # outputs = model.generate(**inputs, max_new_tokens=len(sentence_text))
    outputs = model.generate(**inputs, max_new_tokens=len(sentence_text), do_sample=True, top_k=30, top_p=0.95, num_return_sequences=1) # for translation, https://huggingface.co/docs/transformers/tasks/translation#inference

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if remove_input:
        # 從 generate 出來的 output 中刪除 input text 的部分
        cleaned_output = decoded_output.replace(full_prompt, "")
        return cleaned_output
    else:
        return decoded_output

sentence = "我国古代品评书画艺术的三个等级，即神品、妙品、能品。"
sentence1 = "台中后里位在台湾南北的交会点，隐藏着许多全国知名的景点。"
sentence2 = "我干什么不干你事。"
sentence3 = "我发现太后的头发很干燥。"
sentence4 = "芋头发芽了。"
sentence5 = "再坐在电脑前面 我头发都没了T_T。"
sentence6 = "他觉得丑时人生的通常都比较丑。"
sentence7 = "而在中时报系以不堪亏损为由舍弃晚报的同时，另方面却持续入股中天电视台，并有意在未来收购中视，成就跨媒体集团霸业。"

translate_prompt_template = """
### Instruction:
翻譯成繁體中文: {}
### Response:
"""
device = "cuda:0"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_model_path,
    local_files_only=True, 
    load_in_4bit=True, 
    # device_map={"": "cpu"}
)
# model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, load_in_4bit=True, device_map="auto")

print(get_response(model, translate_prompt_template, sentence))
print('---')
print(get_response(model, translate_prompt_template, sentence1))
print('---')
print(get_response(model, translate_prompt_template, sentence2))
print('---')
print(get_response(model, translate_prompt_template, sentence3))
print('---')
print(get_response(model, translate_prompt_template, sentence4))
print('---')
print(get_response(model, translate_prompt_template, sentence5))
print('---')
print(get_response(model, translate_prompt_template, sentence6))
print('---')
print(get_response(model, translate_prompt_template, sentence7))
print('---')
