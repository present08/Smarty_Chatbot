import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing

# 의도 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, preprocess):


        self.labels = {0: "위치", 1: "번호", 2: "가격", 3: "영업시간", 4: "대여", 5:"반납", 6:"수강", 7: "주차", 8: "예약", 9: "기타"}
        self.model = load_model(model_name)
        self.p = preprocess


    def predict_class(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        from config.GlobalParams import MAX_SEQ_LEN
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]