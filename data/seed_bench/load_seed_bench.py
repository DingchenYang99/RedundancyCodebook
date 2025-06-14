import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    source_path = '/DATA2/yangdingchen/SEED/'
    
    file_path = source_path + 'data/'
    image_root = source_path + 'images/'
    file_names = [fi for fi in sorted(os.listdir(file_path)) if fi.endswith(".parquet")]
    
    target_type = 'image'
    paths = source_path + f'seed_bench_{target_type}-17kt-local.jsonl'  # TODO
    print(paths)
    if os.path.exists(paths):
        print(paths+" already exists")
        raise ValueError
    
    fs_paths = open(paths, 'w', encoding="utf-8")
    cnt = 0
    for file_name in tqdm(file_names):
        
        df = pd.read_parquet(os.path.join(file_path, file_name))
        
        for r_idx, r in tqdm(df.iterrows()):
            r_dict = r.to_dict()
            data_type = r_dict["data_type"]
            if data_type != target_type:
                continue
            
            answer = r_dict["answer"]
            choice_a = r_dict["choice_a"]
            choice_b = r_dict["choice_b"]
            choice_c = r_dict["choice_c"]
            choice_d = r_dict["choice_d"]
            question = r_dict["question"]
            question_id = r_dict["question_id"]
            data_id = r_dict["data_id"]
            question_type_id = r_dict["question_type_id"]
            
            sample_id = f'seed_{data_type}_{data_id}-{question_id}'
            raw_image = r_dict["image"].tolist()
            # print(type(raw_image))
            if data_type == 'image':
                raw_image = raw_image[0]
            else:
                raise NotImplementedError
            # try:
            image_byte = raw_image["bytes"]
            image_filename = raw_image["path"] + '.png'
            # assert image_filename == image_id
            # print(image_filename)
            assert image_filename.split(".")[-1] in ["jpg", "jpeg", "png", "webp"], f"path: {image_filename} is not valid"
            
            image_name = image_root + image_filename
            if not os.path.exists(image_name):
                image = Image.open(io.BytesIO(image_byte))
                try:
                    image.save(image_name)
                except:
                    image = image.convert('L')  # for CMYK
                    image.save(image_name)
            assert os.path.exists(image_name)
            # except:
            #     print(f"{sample_id} image have less than four")
            #     continue
            
            datasample = {
                "sample_id":sample_id,
                "image_name":image_name,
                "question":question,
                "answer":answer,
                "choice_a":choice_a,
                "choice_b":choice_b,
                "choice_c":choice_c,
                "choice_d":choice_d,
                "data_type":data_type,
                "data_id":data_id,
                "question_id":question_id,
                "question_type_id":question_type_id,
            }
            # print(datasample)
            cnt += 1
            fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
            fs_paths.flush()
            
            # if cnt == 10:  # TODO
            #     break
            
    fs_paths.close()
    print(f"finished loading {cnt} valid samples")