import pymysql
from config.DatabaseConfig import *
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time

from utils.Preprocess import Preprocess
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin', user_dic='../../utils/user_dic.tsv')

file = '../../train_tools/qna/train_data.xlsx'
df = pd.read_excel(file)

df_data = list(df["질문(Query)"])
sentence_list = []
word_tensor = {}
for i in tqdm(range(len(df_data))):
    word = df_data[i]
    pos = p.pos(word)
    keyword = p.get_keywords(pos)
    sentence = ""
    for k in keyword:
        encode = model.encode(k[0])
        tensor = torch.tensor(encode)
        sentence += k[0]
        word_tensor[k[0]] = tensor
    encode = model.encode(sentence)
    tensor = torch.tensor(encode)
    sentence_list.append(tensor)
    time.sleep(0.1)

db = None
try:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    qna_table_sql = '''
        insert into chatbot_qna(intent, query, answer) value(%s, %s, %s)
    '''
    tensor_table_sql = '''
        insert into chatbot_tensor(data_id, tensor) value(%s, %s)
    '''
    word_tensor_sql = '''
        insert into chatbot_word_tensor(word, tensor) value(%s, %s)
    '''

    with db.cursor() as cursor:
        for idx, row in df.iterrows():
            intent = row['의도(Intent)']
            query = row['질문(Query)'].replace(" ","")
            print("추가 되는 질문 : ", query)
            answer = row['답변(Answer)']
            cursor.execute(qna_table_sql, (intent, query, answer))
            last_inserted_id = cursor.lastrowid
            cursor.execute(tensor_table_sql,(last_inserted_id, sentence_list[idx]))

        for word, tensor in word_tensor.items():
            cursor.execute(word_tensor_sql, (word, tensor))

except Exception as e:
    print(e)

finally:
    db.commit()
    if db is not None:
        db.close()
