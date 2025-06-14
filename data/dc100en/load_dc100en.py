import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    source_path = '/DATA2/yangdingchen/DC100_EN/'
    
    file_path = source_path + 'data/'  # TODO
    image_root = source_path + 'images/'
    file_names = [fi for fi in sorted(os.listdir(file_path)) if fi.endswith(".parquet")]
    
    paths = source_path + 'dc100en-100t-local.jsonl'  # TODO
    print(paths)
    if os.path.exists(paths):
        print(paths+" already exists")
        raise ValueError
    
    fs_paths = open(paths, 'w', encoding="utf-8")
    cnt = 0
    for file_name in tqdm(file_names):
        file_id = '-'.join(file_name.split("-")[:2])
        df = pd.read_parquet(os.path.join(file_path, file_name))
        
        for r_idx, r in tqdm(df.iterrows()):
            
            r_dict = r.to_dict()
            question_id = r_dict["question_id"]
            question = r_dict["question"]
            category = r_dict["category"]
            
            sample_id = f'dc100en_{question_id}'
            raw_image = r_dict["image"]
            # print('path is ' + raw_image["path"])
            image_byte = raw_image["bytes"]
            # image_filename = raw_image["path"]
            image_filename = sample_id + '.png'
            # print(image_filename)
            assert image_filename.split(".")[-1] in ["jpg", "jpeg", "png", "webp"]
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
                "question_id":question_id,
                "image_name":image_name,
                "question":question,
                "category":category,
            }
            # print(datasample)
            cnt += 1
            fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
            fs_paths.flush()
            
            # if cnt == 10:  # TODO
            #     break
            
    fs_paths.close()
    print(f"finished loading {cnt} valid samples")