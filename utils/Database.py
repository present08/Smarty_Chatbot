import pymysql
import pymysql.cursors
import logging

class Database :
    def __init__(self, host, user, password, db_name, charset='utf8'):
        self.host = host
        self.user = user
        self.password = password
        self.charset = charset
        self.db_name = db_name
        self.conn = None

    def connect(self):
        if self.conn != None :
            return
        try:
            self.conn = pymysql.connect(
                host = self.host,
                user = self.user,
                password = self.password,
                db = self.db_name,
                charset = self.charset
            )
            print("DB connected...")
            return self.conn
        except:
            print("Error DB connect")
            return None

    def close(self):
        if self.conn is None:
            return

        if not self.conn.open:
            self.conn = None
            return
        self.conn.close()
        self.conn = None

    def execute(self, sql):
        last_row_id = -1
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
            last_row_id = cursor.lastrowid
            logging.debug("execute last_row_id: %d", last_row_id)
        except Exception as ex:
            logging.error(ex)
        finally:
            return last_row_id

    def select_one(self, sql):
        result = None
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result = cursor.fetchone()
        except Exception as ex:
            logging.error(ex)
        finally:
            return result

    def select_all(self, sql):
        result = None
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
        except Exception as ex:
            logging.error(ex)
        finally:
            return result

    def make_query(self, column, table):
        sql = "select {} from {}".format(column, table)
        return sql

    def load_data(self, intent_name):
        if intent_name != None:
            sql = "select r.id,r.intent, r.query, r.answer, t.tensor from chatbot_qna r join chatbot_tensor t on r.id = t.data_id where r.intent = '{}'".format(
            intent_name)
        else : sql = "select r.id,r.intent, r.query, r.answer, t.tensor from chatbot_qna r join chatbot_tensor t on r.id = t.data_id"
        result = self.select_all(sql)
        return result

    def insert_data(self, intent, query, new_answer, tensor_data):
        sql = "insert into chatbot_qna(intent, query, answer) values(%s, %s, %s)"
        sql_insert_tensor = "INSERT INTO chatbot_tensor (data_id, tensor) VALUES (%s, %s)"
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql,(intent, query, new_answer))
                last_id = cursor.lastrowid
                cursor.execute(sql_insert_tensor, (last_id, tensor_data))
                self.conn.commit()
                print(f"새로운 데이터 등록 id={last_id}, query={query}, intent={intent}, answer={new_answer}")
        except Exception as ex:
            logging.error(ex)

    def insert_word(self, word_tensor):
        for key, value in word_tensor.items():
            sql = 'insert into chatbot_word_tensor(word, tensor) values(%s, %s)'
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute(sql,(key,value))
                    self.conn.commit()
            except Exception as ex:
                print(ex)

    def facility_info(self,keywords):
        sql = 'select * from facility'
        facility_list = self.select_all(sql)
        try:
            for item in keywords:
                for facility in facility_list:
                    if item in facility.values():
                        facility_class = self.select_all("select * from class where facility_id = '{}'".format(facility["facility_id"]))
                        facility_product = self.select_all("select * from product where facility_id = '{}'".format(facility["facility_id"]))
                        return facility, facility_class, facility_product
                    else :
                        facility = None
                        facility_class = None
                        facility_product = None
                        return facility, facility_class, facility_product
        except :
            facility = None
            facility_class = None
            facility_product = None
            return facility, facility_class, facility_product
