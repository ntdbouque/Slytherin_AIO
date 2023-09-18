'''
    Author: Nguyễn Trường Duy
    Purpose: read any json file
    Date: 3/9/2023
'''
'''
    Author: Nguyễn Trường Duy
    Purpose: read any file .json
    Date: 2/9
'''


import json
import glob

with open('/home/duy/Documents/Competition/Video-Text-Retrieval/objects-b1/objects/L01_V001/0003.json', 'r') as data:
    json_file = json.load(data)

for id, value in json_file.items():
    print(f'id: {id}, value: {value}')
    print("*****************")
    