'''
    Author: Nguyễn Trường Duy
    Purpose: text search and image search base on clip model vitb16 and clip
    feature
    Date: 3/9/2023
'''


import numpy as np
import faiss
#import glob
import json
import matplotlib.pyplot as plt
#import os
import math
from nlp_processing import Translation
import clip
import torch
#import pandas as pd
#import re
from langdetect import detect

class MyFaiss:
    def __init__(self, root_database: str, bin_file:str, json_path:str):
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = self.load_json_file(json_path)

        self.translater = Translation()

        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, preprocess = clip.load('ViT-B/16', device = self.__device)

    def load_json_file(self, json_path:str):
        js = json.load(open(json_path, 'r'))

        return {int(k):v for k,v in js.items()}
    
    def load_bin_file(self, bin_file:str):
        return faiss.read_index(bin_file)
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows +1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")
        
        plt.show()


    def image_search(self, id_query, k):
        query__feats = self.index.reconstruct(id_query).reshape(1,-1)

        scores, idx_image = self.index.search(query__feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]  # keyframes_id.json
        # image_paths = [info for info in infos_query] # dict_id2img_path.json

        return scores, idx_image, infos_query, image_paths
    
    def text_search(self,text, k):
        if detect(text) == 'vi':
            text = self.translater(text)

        text = clip.tokenize([text]).to(self.__device)  
        text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]  # keyframes_id.json
        #image_paths = [info for info in infos_query] #dict_id2img_path.json

        return scores, idx_image, infos_query, image_paths
    
def main():
    bin_file='/home/duy/Documents/Competition/AIC2023/dict/faiss_cosine.bin'
    json_path = '/home/duy/Documents/Competition/AIC2023/dict/keyframes_id(sai).json'

    cosine_faiss = MyFaiss('/home/duy/Documents/Competition/AIC2023/Database', bin_file, json_path)
        
    #text = 'trên khung hình có nhiều người dân tộc đang nhảy múa, quần áo của họ rất sặc sỡ, có cả phụ nữ, trẻ em, đàn ông, họ đứng xoay quanh với nhau thành hình tròn'
    id_query = 201

    # scores, idx_image, infos_query, image_paths = cosine_faiss.text_search(text, k=9) # text_search
    scores, idx_image, infos_query, image_paths = cosine_faiss.image_search(id_query, k = 10) # image_search

    
    # print(f'scores: {scores}, idx_image: {idx_image}, infos_query: {infos_query}')
    print(image_paths)

if __name__ ==  '__main__':
    main()
    

