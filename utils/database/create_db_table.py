import pymysql
from config.DatabaseConfig import *

db = None
try:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )


    qnd_table_sql = '''
        create table chatbot_qna(
            id int not null primary key auto_increment,
            intent varchar(50) not null,
            query varchar(1000) not null,
            answer varchar(1000) not null,
            answer_img varchar(1000)
        );
    '''
    tensor_data_sql = '''
    create table chatbot_tensor(
	    id int primary key auto_increment,
        data_id int,
        tensor blob,
        foreign key (data_id) references chatbot_qna(id)
    );
    '''
    word_tensor_sql = '''
    create table chatbot_word_tensor(
        id int primary key auto_increment,
        word varchar(1000),
        tensor blob
    );
    '''

    with db.cursor() as cursor:
        cursor.execute(qnd_table_sql)
        cursor.execute(tensor_data_sql)
        cursor.execute(word_tensor_sql)

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
