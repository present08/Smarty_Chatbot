import datetime

import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util

class FindAnswer:
    def __init__(self, preprocess, db):
        self.p = preprocess

        self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

        self.db = db

    def search(self, query, intent):
        # 형태소 분석
        pos = self.p.pos(query)

        # 불용어 제거
        keywords = self.p.get_keywords(pos, without_tag=True)
        query_pre = ""
        for k in keywords:
            query_pre += str(k)

        print("단어 분석 : ",keywords)
        query_encode = self.model.encode(query_pre)
        query_tensor = torch.tensor(query_encode)

        tensor_list = []
        db_data = self.db.load_data(intent)
        for i in db_data:
            tensor_string = i.get('tensor').decode('utf-8').strip('tensor[([])]').replace(' ', '').replace('\n', '')
            tensor = torch.tensor([float(x) for x in tensor_string.split(',')])
            tensor_list.append(tensor)
        tensor_query_data = torch.stack(tensor_list)


        cos_sim = util.cos_sim(query_tensor, tensor_query_data)
        best_sim_idx = int(np.argmax(cos_sim))
        selected_qes = [item['query'] for item in db_data][best_sim_idx]
        print("질문 의도 : ", intent)
        print("Load Data Array Size : ",len(db_data)," 가장 높은 유사도를 가진 Index : ", best_sim_idx)


        result = self.db.facility_info(keywords)
        if result is not None:
            facility, facility_class, facility_product = result
        else:
            facility, facility_class, facility_product = None, None, None
        print("facility: ", facility)
        print("facility_class: ", facility_class)
        print("facility_product: ", facility_product)


        if [item['intent'] for item in db_data][best_sim_idx] == intent:
            selected_qes_encode = self.model.encode(selected_qes)
            score = dot(query_tensor, selected_qes_encode) / (norm(query_tensor) * norm(selected_qes_encode))

            if facility is not None :
                if intent == '번호':
                    answer = ([item['answer'] for item in db_data][best_sim_idx]
                              .format(facilities_phone=facility["contact"]))
                elif intent == '가격':
                    answer = ([item['answer'] for item in db_data][best_sim_idx]
                              .format(default_time=facility["default_time"], basic_fee=facility["basic_fee"]))
                elif intent == '영업시간':
                    answer = ([item['answer'] for item in db_data][best_sim_idx]
                              .format(open_time=facility["open_time"], close_time=facility["close_time"]))
                    print(answer + ("-" * 50))
                elif intent == '대여':
                    answer = [item['answer'] for item in db_data][best_sim_idx]
                    for idx, item in enumerate(facility_product, start=1):
                        answer = answer + f"[ 대여 가능 품목 : {item['product_name']}, 대여요금 : {item['price']} 원 ]"
                        if idx < len(facility_product):
                            answer += ",\n"
                        else:
                            answer += " 입니다.\n\n 오늘도 Smarty와 함께 좋은하루되세요 :)"
                elif intent == '수강':
                    now = datetime.date.today()
                    answer = ([item['answer'] for item in db_data][best_sim_idx]).format(now=now)
                    if len(facility_class) > 0:
                        for idx, item in enumerate(facility_class, start=1):

                            if item['start_date'] > now:
                                answer += f"\n  - 강의명 : {item['class_name']}\n  - 수강료 : {item['price']} 원\n  - 수강기간 : {item['start_date']} ~ {item['end_date']}\n\n"
                                if idx < len(facility_class):
                                    answer += ",\n"
                                else:
                                    answer += "상세 강의 정보는 상세 페이지 참조바랍니다. \n\n 오늘도 Smarty와 함께 좋은하루되세요 :)"
                            else : answer += "현재 수강 가능한 강의가 없습니다. 다음에 이용해 주세요. \n\n 오늘도 Smarty와 함께 좋은하루되세요 :)"
                    else :
                        answer += "현재 수강 가능한 강의가 없습니다. 다음에 이용해 주세요. \n\n 오늘도 Smarty와 함께 좋은하루되세요 :)"
                elif intent == '예약':
                    answer = ([item['answer'] for item in db_data][best_sim_idx]
                              .format(facilities_phone=facility["contact"], opentime=facility["open_time"],
                                      closetime=facility["close_time"]))
                else :
                    # intent = 위치, 주차, 문의, 반납
                    answer = ([item['answer'] for item in db_data][best_sim_idx])
                # imageUrl = [item['answer_img'] for item in db_data][best_sim_idx]
                set_answer = [item['answer'] for item in db_data][best_sim_idx]
            else:
                answer = "해당 시설은 없는 시설입니다. 관리자에 문의 바랍니다.\n\n 오늘도 Smarty와 함께 좋은하루되세요 :)"
                # imageUrl = [item['answer_img'] for item in db_data][best_sim_idx]
                set_answer = [item['answer'] for item in db_data][best_sim_idx]
        else:
            score = 0
            answer = [item['answer'] for item in db_data][best_sim_idx]
            set_answer = [item['answer'] for item in db_data][best_sim_idx]
            # imageUrl = [item['answer_img'] for item in db_data][best_sim_idx]

        return selected_qes,score, answer, keywords, query_tensor, set_answer

    def search_again(self, query):
        query_encode = self.model.encode(query)
        query_tensor = torch.tensor(query_encode)

        tensor_list = []
        db_data = self.db.load_data(None)
        for i in db_data:
            tensor_string = i.get('tensor').decode('utf-8').strip('tensor[([])]').replace(' ', '').replace('\n', '')
            tensor = torch.tensor([float(x) for x in tensor_string.split(',')])
            tensor_list.append(tensor)
        tensor_query_data = torch.stack(tensor_list)

        # 코사인 유사도를 통해 질문 데이터 선택
        cos_sim = util.cos_sim(query_tensor, tensor_query_data)
        best_sim_idx = int(np.argmax(cos_sim))
        selected_qes = [item['query'] for item in db_data][best_sim_idx]
        tensor_data = tensor_query_data[best_sim_idx]

        # 선택된 질문 문장 인코딩
        selected_qes_encode = self.model.encode(selected_qes)

        # 유사도 점수 측정
        score = dot(query_tensor, selected_qes_encode) / (norm(query_tensor) * norm(selected_qes_encode))

        # 답변
        answer = [item['answer'] for item in db_data][best_sim_idx]
        intent = [item['intent'] for item in db_data][best_sim_idx]
        # imageUrl = [item['answer_img'] for item in db_data][best_sim_idx]

        return score, answer, intent, tensor_data