import threading
import json
import tensorflow as tf

from config.DatabaseConfig import *
from utils.BotServer import BotServer
from utils.Database import Database
from utils.FindAnswer import FindAnswer
from models.intent.IntentModel import IntentModel
from utils.Preprocess import Preprocess

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# 로그 기능 구현
from logging import handlers
import logging

#log settings
LogFormatter = logging.Formatter('%(asctime)s,%(message)s')

#handler settings
LogHandler = handlers.TimedRotatingFileHandler(filename='./logs/chatbot.log', when='midnight', interval=1, encoding='utf-8')
LogHandler.setFormatter(LogFormatter)
LogHandler.suffix = "%Y%m%d"

#logger set
Logger = logging.getLogger()
Logger.setLevel(logging.ERROR)
Logger.addHandler(LogHandler)

# DataBase
db = Database(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME)
p, intent= None, None
# 전처리 객체 생성
try:
    p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin', user_dic='utils/user_dic.tsv')
    print("Preprocess Load completed...")
except: print("Preprocess failed...")

# 의도 파악 모델
try:
    intent = IntentModel(model_name='./models/intent/intent_model.h5', preprocess=p)
    print("Intent Model Load completed...")
except: print("Intent Model failed...")

def to_client(conn, addr, params):
    db = params['db']
    try:
        db.connect()  # 디비 연결

        # 데이터 수신
        read = conn.recv(2048) # 수신 데이터가 있을 때까지 블로킹
        print('======================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            print('클라이언트 연결 끊어짐')
            exit(0)  # 스레드 강제 종료

        recv_json_data = json.loads(read.decode())
        query = recv_json_data['Query'].replace(" ","")

        intent_pred = intent.predict_class(query)
        intent_name = intent.labels[intent_pred]

        f = FindAnswer(db=db, preprocess=p)
        selected_qes, score, answer, keywords, query_tensor, set_answer = f.search(query, intent_name)
        print("키워드 : ", keywords)
        print("질문 : ",query)
        print("질문 의도 : ", intent_name)
        print("가장 가까운 유사도 점수 : ",score)
        print("답변 : ",answer)


        if score < 0.7:
            answer = "부정확한 질문이거나 답변할 수 없습니다.\n 수일 내로 답변을 업데이트하겠습니다.\n 죄송합니다 :("
            # imageUrl = "없음"
            # 사용자 질문, 예측 의도, 선택된 질문, 선택된 질문 의도, 유사도 점수
            Logger.addHandler(LogHandler)
            Logger.error(f"{query},{intent_name},{selected_qes},{score}")
        else :
            if score > 0.83:
                db.insert_data(intent_name, query, set_answer, query_tensor)

        send_json_data_str = {
            "Query": selected_qes,
            "Answer": answer,
            # "imageUrl": imageUrl,
            "Intent": intent_name
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode()) # 응답 전송

    except Exception as ex:
        print(ex)

    finally:
        if db is not None:
            db.close()
            print("db close...")
        conn.close()


if __name__ == '__main__':
    port = 5050
    listen = 1000
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start..")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            "db": db
        }
        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
            params
        ))
        client.start()
