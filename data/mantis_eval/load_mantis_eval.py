import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    source_path = '/DATA2/yangdingchen/Mantis-Eval/mantis_eval/'
    
    image_root = source_path + 'images/'
    file_name = 'test-00000-of-00001.parquet'
    
    paths = source_path + 'mantis_eval_test_217t-local.jsonl'  # TODO
    print(paths)
    if os.path.exists(paths):
        print(paths+" already exists")
        raise ValueError
    
    fs_paths = open(paths, 'w', encoding="utf-8")
    cnt = 0
    df = pd.read_parquet(os.path.join(source_path, file_name))
    
    for r_idx, r in tqdm(df.iterrows()):
        r_dict = r.to_dict()
        # print(r_dict.keys())
        # print(r_dict["texts"])  # [{"user":,"assistant":,"source":}]
        # print(len(r_dict["images"]))  # 2
        # print(r_dict["images"][0].keys())  # dict_keys(['bytes', 'path'])
        # print(r_dict["images"][-1]["path"])  # None
        # break
        sample_id = r_dict["id"]
        question_type = r_dict["question_type"]
        question = r_dict["question"]
        options = r_dict["options"].tolist()
        answer = r_dict["answer"]
        category = r_dict["category"]
        
        raw_images = r_dict["images"]
        assert len(raw_images) >= 2
        
        image_path_list = []
        for image_id, image_sample in enumerate(raw_images):
            image_byte = image_sample["bytes"]
            # image_name = image_sample["path"]
            image_name = f"{sample_id}_{image_id}.jpg"
            assert image_name.split(".")[-1] in ["jpeg", "jpg", "png"]
            image_path = image_root + image_name
            image = Image.open(io.BytesIO(image_byte))
            image.save(image_path)
            assert os.path.exists(image_path)
            image_path_list.append(image_path)
        
        datasample = {
            "sample_id":sample_id,
            "image_names":image_path_list,
            "question":question,
            "options":options,
            "answer":answer,
            "category":category,
            "question_type":question_type,
        }
        # print(datasample)
        cnt += 1
        fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
        fs_paths.flush()
        
        # if cnt == 10:  # TODO
        #     break
            
    fs_paths.close()
    print(f"finished loading {cnt} valid samples")