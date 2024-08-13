import pandas as pd
import nltk.translate.gleu_score as gleu
from nltk.translate.bleu_score import sentence_bleu

import opencc

t2s_converter = opencc.OpenCC('t2s.json')

"""traditional chinese to simplified chinese"""
def convert_TC_to_SC(zhtw_text: str, converter=None):
    if not converter:
        converter = opencc.OpenCC('t2s.json')
    return converter.convert(zhtw_text)


def evaluate_bleu_gleu(df, output_csv_path):
    scores_list = []

    for _, row in df.iterrows():
        ref = list(row['ch'])
        ch_xmucc_src = list(row['ch_xmucc'])

        scores = {
            'ch': row['ch'],
            'ch_SC': row['ch_SC'],
            'ch_xmucc': row['ch_xmucc'],
            'xmucc_and_truth_bleu_score': sentence_bleu([ref], ch_xmucc_src),
            'xmucc_and_truth_gleu_score': gleu.sentence_gleu([ref], ch_xmucc_src, min_len=1, max_len=2),
        }
        scores_list.append(scores)

    results_df = pd.DataFrame(scores_list)
    print(results_df)
    results_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    src_txt_path = './exp_outputs/raw_inference_results/xmucc_translate_result.txt'
    output_csv_path = 'xmucc_evaluation_results_with_score.csv'

    df = pd.read_csv('test_dataset.csv')
    res = pd.read_csv(src_txt_path, header=None, names=['ch_xmucc'])

    # res['ch_SC_xmucc'] = list(map(convert_TC_to_SC, res['ch_xmucc']))
    # merged_df = pd.merge(df, res, left_on='ch_SC', right_on='ch_SC_xmucc', how='inner')

    merged_df = pd.concat([df, res], axis=1) # 用 index join
    print(merged_df.loc[0, 'ch'], merged_df.loc[0, 'ch_xmucc'])

    evaluate_bleu_gleu(merged_df, output_csv_path)
    # print(res)
