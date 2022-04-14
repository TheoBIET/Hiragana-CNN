import os,struct

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from PIL import Image
from utils.constants import HIRAGANA

BASE_PATH = './data/etl7'
UNPACK_STRING = '>H2sH6BI4H4B4x2016s4x'
FILES = ['ETL7LC_1']
RECORDS_NB = [9600, 7200, 9600, 7200]
RECORD_LENGTH = 2052 # bytes
WIDTH = 64
HEIGHT = 63
COLUMNS_LABELS = ['hiragana','image_data']

records_list = []

for filename in FILES:
    file_index = FILES.index(filename)
    records_nb = RECORDS_NB[file_index]
    count = 0
    f = open(f'{BASE_PATH}/{filename}', 'rb')

    while count <= records_nb:
        record_string = f.read(RECORD_LENGTH)

        if len(record_string) < RECORD_LENGTH:
            break
            
        record = struct.unpack(UNPACK_STRING, record_string)
        
        phonetic = record[1].decode('ascii')
        hiragana = HIRAGANA[phonetic]
        
        record_data = [hiragana,record[18]]
        
        records_list.append(record_data)     
        count+=1

    f.close()
    
df = pd.DataFrame(records_list, columns=COLUMNS_LABELS)
df.head()

size = len(HIRAGANA) / 2
hiraganas = list(HIRAGANA.values())

for H in hiraganas:
    i = hiraganas.index(H)
    char, img_data = df[df['hiragana'] == H].values[0]
    img = np.array(Image.frombytes('F', (WIDTH, HEIGHT), img_data, 'bit', 4))
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

reshaped_arr = np.reshape(np.array(Image.frombytes('F', (WIDTH, HEIGHT), img_data, 'bit', 4)), HEIGHT*WIDTH)