import torch
import numpy as np
from sentence_transformers import SentenceTransformer,util
from utils.FindAnswer import FindAnswer

class DataUpdate:
    def __init__(self, preprocess, db):
        # 챗봇 텍스트 전처리기
        self.p = preprocess

        self.db = db

        # pre-trained SBERT
        self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

        # Word Tensor Load
        sql = db.make_query("tensor", "word_tensor")
        tensor_list = []
        rows = db.select_all(sql)
        # print(rows)
        for i in rows:
            tensor_string = i.get('tensor').decode('utf-8').strip('tensor[([])]').replace(' ', '').replace('\n', '')
            tensor = torch.tensor([float(x) for x in tensor_string.split(',')])
            tensor_list.append(tensor)

        self.tensor_word_data = torch.stack(tensor_list)

        # word Data 추출
        sql = self.db.make_query("word", "word_tensor")
        self.db_word = [i.get('word') for i in self.db.select_all(sql)]

    def update_data(self, keywords, query):
        print("keywords : " , keywords)
        print("업데이트 예정 Query : " , query)
        query_data = []
        word_filter = {}
        # DB 단어 추가
        for i in keywords:
            try:
                if self.db_word.index(i):
                    pass
            except ValueError:
                i_encode = self.model.encode(i)
                i_tensor = torch.tensor(i_encode)
                word_filter[i] = i_tensor
        self.db.insert_word(word_filter)

        # 형태소화 된 질문(keywords)을 텐서화
        for i in keywords:
            i_encode = self.model.encode(i)
            i_tensor = torch.tensor(i_encode)
            query_data.append(i_tensor)
        # [word] 텐서화 된 형태소와 데이터 형태소 비교
        word_to_sentence = ""
        for i in query_data:
            # 텐서화 된 형태소 비교
            # keyword tensor와  word_data tensor 비교
            cos_sim = util.cos_sim(i, self.tensor_word_data)
            # 비슷한 형태소 추출
            best_sim = int(np.argmax(cos_sim))
            print("best_sim: ", cos_sim[0][best_sim])

            # 가장 근접한 형태소 수치가 0.7 이상이면 문자열에 추가
            if cos_sim[0][best_sim] > 0.75:
                word_to_sentence += self.db_word[best_sim]
            print("DB_word: ", best_sim , self.db_word[best_sim])
        print("신규 질문", word_to_sentence)



        # 새로 만들어진 질문으로 다시 검색
        f = FindAnswer(db=self.db, preprocess=self.p)
        score, answer, intent, tensor_data = f.search_again(word_to_sentence)

        if score > 0.7:
            self.db.insert_data(intent, query, answer, tensor_data)
            return answer
        else :
            return None

