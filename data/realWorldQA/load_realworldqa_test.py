import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    source_path = '/DATA2/yangdingchen/RealWorldQA/'
    
    file_path = source_path + 'data/'
    image_root = source_path + 'images/'
    file_names = [fi for fi in sorted(os.listdir(file_path)) if fi.endswith(".parquet")]
    
    paths = source_path + 'realWorldQA_test-765t-local.jsonl'  # TODO
    print(paths)
    if os.path.exists(paths):
        print(paths+" already exists")
        raise ValueError
    
    fs_paths = open(paths, 'w', encoding="utf-8")
    cnt = 0
    for file_name in tqdm(file_names):
        file_id = file_name.split("-")[1]
        df = pd.read_parquet(os.path.join(file_path, file_name))
        
        for r_idx, r in tqdm(df.iterrows()):
            
            r_dict = r.to_dict()
            answer = r_dict["answer"]
            question = r_dict["question"]
            image_filename = r_dict["image_path"]
            assert image_filename.split(".")[-1] in ["jpg", "jpeg", "png", "webp"]
            
            sample_id = f'realworldqa_test_{file_id}_{r_idx}'
            raw_image = r_dict["image"]
            
            image_byte = raw_image["bytes"]
            # image_filename = raw_image["path"]
            # assert image_filename == image_id
            # print(image_filename)
            
            image_name = image_root + image_filename
            image = Image.open(io.BytesIO(image_byte))
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
            }
            # print(datasample)
            cnt += 1
            fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
            fs_paths.flush()
            
            # if cnt == 10:  # TODO
            #     break
            
    fs_paths.close()
    print(f"finished loading {cnt} valid samples")