import sys 
import os 
import sqlite3
import re
import datetime
import hashlib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
#----------------------------------------------------------------------------------------------------------------------------------
def text_cleaner(text):
    text = text.lower() # ���������� � lowercase,
    
    text = re.sub( r'https?://[\S]+', ' url ', text) # ������ �������� ������
    text = re.sub( r'[\w\./]+\.[a-z]+', ' url ', text) 
 
    text = re.sub( r'\d+[-/\.]\d+[-/\.]\d+', ' date ', text) # ������ ���� � �������
    text = re.sub( r'\d+ ?��?', ' date ', text) 
    text = re.sub( r'\d+:\d+(:\d+)?', ' time ', text) 

    # text = re.sub( r'@\w+', ' tname ', text ) # ������ ��� twiter
    # text = re.sub( r'#\w+', ' htag ', text ) # ������ ��������


    
    stw = ['�', '��', '��', '��', '�', '���', '��', '��', '��', '���', '���', '��',
           'a', 'at', 'on', 'of', 'and', 'or', 'in', 'for', 'at' ]
    text = re.sub( r'<[^>]*>', ' ', text) # �������� html �����
    text = re.sub( r'[\W]+', ' ', text ) # �������� ������ ��������

    return  text
#------------------------------------------------------------------------------------------------------------------------------------
def load_data():
    dbname = 'data/export_file utf-8.sqlite'
    data = { 'IE_DETAIL_TEXT':[],'IE_NAME':[], 'tag':[] }
    conn = sqlite3.connect(dbname)
    try:
        c = conn.cursor()
        for row in c.execute('SELECT * FROM data'):
            data['IE_DETAIL_TEXT'] += [row[1]]
            data['IE_NAME'] += [row[2]]
            data['tag'] += [row[2]]
    finally:
        conn.close()
    return data
#-------------------------------------------------------------------------------------------------------------------------------------
def save2db(data):
    load_data()
    dbname = 'result/export file utf-8 with clasters.sqlite'
    conn = sqlite3.connect(dbname)
    try:
        c = conn.cursor()
        c.execute("CREATE TABLE data(IE_XML_ID TEXT PRIMARY KEY, IE_DETAIL_TEXT TEXT, IE_NAME TEXT, tag TEXT)")
        data1 ={'tag':[]}
        for n in range(0,len(data['IE_NAME'])):
            t = str(data['IE_DETAIL_TEXT'][n]) + str(data['IE_NAME'][n]) + str(data['tag'][n]) + str(datetime.datetime.now())
            rec_hash = hashlib.sha256(t.encode('utf-8')).hexdigest()
            c.execute("INSERT INTO data VALUES (?, ?, ?, ?)",  (rec_hash, str(data['IE_DETAIL_TEXT'][n]),str(data['IE_NAME'][n]),str(data['tag'][n])  ) ) 
        conn.commit()
    
    finally:
        conn.close()
#-----------------------------------------------------------------------------------------------------------------------------------------
def main():
    print("[i] ��������� ������...")
    data = load_data()
    print("\t�������: ",len(data['IE_NAME']))
    

    print("[i] ������� ������...")
    D = [ text_cleaner(t) for t in data['IE_NAME'] ]

    n_clusters=6
    print("[i] �������� ��������������...")
 
    text_clstz = Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ( 'km', KMeans(n_clusters=n_clusters)),
                    #( 'km', KMeans(n_clusters=n_clusters, init='k-means+', n_init=10, max_iter=300, tol=1e-04, random_state=0) )
                            ])
    text_clstz.fit(D)
    data['tag'] = text_clstz.predict(D)
    print("\t���������� ���������:",len(set(data['tag'])))
    print('\t-----------------')
    print("[i] ��������� ���������...")
    
    save2db(data)
#-------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit( main() )

