import os
import ast
import time
import jieba
import torch
import warnings
import argparse

import opencc
import pandas as pd
from tqdm.auto import tqdm

import nltk.translate.gleu_score as gleu
from nltk.translate.bleu_score import sentence_bleu

from transformers import LlamaTokenizer
from peft import AutoPeftModelForCausalLM

model_name = "/home/wirl/ytc/TAIDE-LX-7B-Chat"
QLORA_DIR = 'qlora3' # TAIDE

translate_prompt_template = """
### Instruction:
翻譯成繁體中文: {}
### Response:
"""

t2s_converter = opencc.OpenCC('t2s.json')

"""simplified chinese to traditional chinese"""
def convert_SC_to_TC(zhcn_text: str, converter=None):
    if not converter:
        converter = opencc.OpenCC('s2t.json')
    return converter.convert(zhcn_text)

def init_model(ft_num):
    adapter_model_path = f"/home/wirl/ytc/要寫的論文研究/code/{QLORA_DIR}/output/checkpoint-{ft_num}"

    print(f"Using ft_num={ft_num} model.")

    # https://github.com/artidoro/qlora/issues/29#issuecomment-1737072311
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_model_path,
        local_files_only=True, 
        load_in_4bit=True, 
    )
    return model

def get_response(tokenizer, model, prompt_template, sentence_text, remove_input=True):
    device = "cuda:0"
    full_prompt = prompt_template.format(sentence_text)

    # temperature, top_p, and top_k are only active when do_sample=True
    # if you set Top-k to 10, the LLM will only consider the 10 most probable next words. 
    # This will result in more fluent text, but it will also reduce the diversity of the text. 
    # TOP_K = 30
    TOP_K = 50
    # TOP_K = 70
    # If you set Top-p to 0.9, the LLM will only generate words that have a probability of at least 0.9. 
    # This will result in more diverse text, but it could also result in less fluent text.
    TOP_P = 0.95
    # TOP_P = 1.0

    # TEMP = 1.0 # don't specify

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # outputs = model.generate(**inputs, max_new_tokens=len(sentence_text))
    outputs = model.generate(
        **inputs, 
        max_new_tokens=len(sentence_text), 
        do_sample=True, 
        # temperature=TEMP,
        top_k=TOP_K, 
        top_p=TOP_P, 
        num_return_sequences=1,
    ) # for translation, https://huggingface.co/docs/transformers/tasks/translation#inference

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if remove_input:
        # 從 generate 出來的 output 中刪除 input text 的部分
        cleaned_output = decoded_output.replace(full_prompt, "")
        return cleaned_output
    else:
        return decoded_output

# baseline_res 與 model_res 長度不同，需要建立在這個前提下做替換，
# 由於 model_res 的幻覺長度通常都會大於應有的長度，因此這邊實作了一個 reverse mapping 的機制將 model_res 的字修正回去
def revise_asymmetric_characters_by_llm(baseline_res, model_res, mappings):
    reverse_mapping = {} # ex. {'系': '系', '係': '系', '繫': '系'...}
    for key, values in mappings.items():
        for value in values:
            reverse_mapping[value] = key

    baseline_res_list = list(baseline_res)
    model_res_list = list(model_res)

    for i, char in enumerate(baseline_res_list):
        if char in reverse_mapping:
            # Condition statements explanations:
            # 1. prevent index error
            # 2. assure model_res is in mappings candidates, 
            #   ex. char='係' -> reverse_mappings['係']='系' -> mappings['系']=['系', '係', '繫']
            # 3. if baseline_res[i] != model_res[i], then replace character
            if i < len(model_res_list) and model_res_list[i] in mappings[reverse_mapping[char]] and model_res_list[i] != char:
                baseline_res_list[i] = model_res_list[i]

    return ''.join(baseline_res_list)

def main():
    parser = argparse.ArgumentParser(description='Use ft_num to specify and generate text.')
    parser.add_argument('--ft_num', type=int, required=True, help='Specify the value for ft_num.') # 1250 is recommended
    args = parser.parse_args()

    RESULT_PATH = "xmucc_inference_result_qlora_1250_TAIDE.csv"
    res = pd.DataFrame(columns=[
        'baseline_response', 
        'model_response', 
        'revised_response',
        'true_sentence', 
        'bl_and_truth_bleu_score',
        'bl_and_truth_gleu_score',
        'model_and_truth_bleu_score',
        'model_and_truth_gleu_score',
        'rev_and_truth_bleu_score',
        'rev_and_truth_gleu_score'
    ])

    if not os.path.isfile(RESULT_PATH):
        res.to_csv(RESULT_PATH, index=False)

    tokenizer = LlamaTokenizer.from_pretrained(model_name, local_files_only=True, legacy=True)
    model = init_model(args.ft_num)

    data = pd.read_csv("簡化字對照標準字總表.csv")
    asy_mappings = data[data['非對稱簡繁字'] == True].set_index('簡體字')['正體字'].apply(ast.literal_eval).to_dict()

    df = pd.read_csv('test_dataset.csv')
    XMUCC_res_df = pd.read_csv('./exp_outputs/raw_inference_results/xmucc_evaluation_results_with_score.csv')

    merged_df = pd.merge(df, XMUCC_res_df,  how='inner', on = ['ch','ch_SC'])

    pb = tqdm(total=len(merged_df), nrows=4, position=0, leave=True)

    # print(merged_df)

    for _, row in merged_df.iterrows():
        org_sentence = row['ch_SC'] # simplified chinese
        true_sentence = row['ch'] # ground truth traditional chinese

        # org_sentence = "而在中时报系以不堪亏损为由舍弃晚报的同时，另方面却持续入股中天电视台，并有意在未来收购中视，成就跨媒体集团霸业。" # simplified chinese
        # true_sentence = "而在中時報系以不堪虧損為由捨棄晚報的同時，另方面卻持續入股中天電視台，並有意在未來收購中視，成就跨媒體集團霸業。" # ground truth traditional chinese
        # baseline_response = convert_SC_to_TC(org_sentence) # OpenCC
        baseline_response = row['ch_xmucc']

        model_response = get_response(tokenizer, model, translate_prompt_template, org_sentence).strip() # LLM
        revised_response = revise_asymmetric_characters_by_llm(baseline_response, model_response, asy_mappings) # revise baseline res by LLM

        ref = list(true_sentence)

        baseline_src = list(baseline_response)
        model_src = list(model_response)
        revised_src = list(revised_response)

        bl_and_truth_bleu_score = sentence_bleu([ref], baseline_src)
        bl_and_truth_gleu_score = gleu.sentence_gleu([ref], baseline_src, min_len=1, max_len=2)

        rev_and_truth_bleu_score = sentence_bleu([ref], revised_src)
        rev_and_truth_gleu_score = gleu.sentence_gleu([ref], revised_src, min_len=1, max_len=2)

        model_and_truth_bleu_score = sentence_bleu([ref], model_src)
        model_and_truth_gleu_score = gleu.sentence_gleu([ref], model_src, min_len=1, max_len=2)

        new_row = {
            'baseline_response': baseline_response,
            'model_response': model_response,
            'revised_response': revised_response,

            'true_sentence': true_sentence,

            'bl_and_truth_bleu_score': bl_and_truth_bleu_score,
            'bl_and_truth_gleu_score': bl_and_truth_gleu_score,
            'model_and_truth_bleu_score': model_and_truth_bleu_score,
            'model_and_truth_gleu_score': model_and_truth_gleu_score,
            'rev_and_truth_bleu_score': rev_and_truth_bleu_score,
            'rev_and_truth_gleu_score': rev_and_truth_gleu_score
        }
        print(f"Now inference :{new_row}")

        res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)

        pb.update(1) # update progress bar
        pb.set_description('Inference process ฅ(*°ω°*ฅ)*... ', refresh=True)

    res.to_csv(RESULT_PATH, index=0)
    print(f"Inference result have been saved to {RESULT_PATH}")


if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time：{execution_time}s")
