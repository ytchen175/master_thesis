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
        ch_google_src = list(row['ch_google'])

        scores = {
            'google_and_truth_bleu_score': sentence_bleu([ref], ch_google_src),
            'google_and_truth_gleu_score': gleu.sentence_gleu([ref], ch_google_src, min_len=1, max_len=2),
        }
        scores_list.append(scores)

    results_df = pd.DataFrame(scores_list)
    print(results_df)
    results_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    src_csv_path = 'google_translate_result.csv'
    output_csv_path = 'google_evaluation_results.csv'

    merged_df = pd.read_csv(src_csv_path)
    # df = pd.read_csv('test_dataset.csv')
    # res = pd.read_csv(src_csv_path)

    # res['ch_SC_google'] = list(map(convert_TC_to_SC, res['ch_google']))

    # merged_df = pd.merge(df, res, left_on='ch_SC', right_on='ch_SC_google', how='inner')
    # # print(merged_df)
    # print(merged_df.loc[0, 'ch'], merged_df.loc[0, 'ch_google'])

    evaluate_bleu_gleu(merged_df, output_csv_path)
