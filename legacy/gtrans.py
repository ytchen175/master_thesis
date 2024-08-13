import os
import time
import csv
import json
import pandas as pd
import googletrans

from pprint import pprint
from tqdm.auto import tqdm

import opencc

t2s_converter = opencc.OpenCC('t2s.json')

"""traditional chinese to simplified chinese"""
def convert_TC_to_SC(zhtw_text: str, converter=None):
    if not converter:
        converter = opencc.OpenCC('t2s.json')
    return converter.convert(zhtw_text)


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def main():
    translator = googletrans.Translator()

    rollback_filename = 'google_translate_result_rollback.csv'
    df = pd.read_csv('test_dataset.csv')

    # sentences = df['ch_SC'].to_list()

    # # 回補資料
    # google_res = pd.read_csv('google_translate_result.csv')
    need_rollback = pd.read_csv('temp.csv')
    # # ch_SC_google = list(map(convert_TC_to_SC, google_res['ch_google']))
    # # sentences = set(df['ch_SC'].to_list()) - set(ch_SC_google)
    # need_rollback = df[~df['ch_SC'].isin(google_res['ch_SC'])]

    header = ['ch_google', 'ch_SC_google']

    pb = tqdm(total=len(need_rollback), nrows=4, position=0, leave=True)

    with open(rollback_filename, 'a+', newline='', encoding='utf-8') as csvfile:
        file_empty = os.stat(rollback_filename).st_size == 0

        csvwriter = csv.writer(csvfile)

        if file_empty:
            csvwriter.writerow(header)

        for _, row in need_rollback.iterrows():
            try:
                translated_sentence = translator.translate(row['ch_SC'], src='zh-cn', dest='zh-tw')
            except:
                print("retry...")
                time.sleep(2)
                continue

            time.sleep(0.3)

            csvwriter.writerow([translated_sentence.text, row['ch_SC']])

            pb.update(1) # update progress bar
            pb.set_description('Google Translate process ฅ(*°ω°*ฅ)*... ', refresh=True)


if __name__ == '__main__':
    main()
  
