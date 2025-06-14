import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    root_path = '/DATA2/yangdingchen/MUIRBENCH/'
    muri_source_path = root_path + 'data/'
    image_save_path = root_path + 'images/'
    
    fs_path = root_path + 'muir_bench_test_2kt-local.jsonl'
    print(fs_path)
    if os.path.exists(fs_path):
        print(fs_path+" already exists")
        raise ValueError
    
    fs_paths = open(fs_path, 'w', encoding="utf-8")
    
    file_names = sorted([q for q in os.listdir(muri_source_path) if q.endswith(".parquet")])
    
    for file_name in tqdm(file_names):
        df = pd.read_parquet(os.path.join(muri_source_path, file_name))
        
        for r_idx, r in tqdm(df.iterrows()):
            r_dict = r.to_dict()
            # print(r_dict.keys())
            sample_id = r_dict["idx"]
            task_name = r_dict["task"]
            image_relation = r_dict["image_relation"]
            image_type = r_dict["image_type"]
            question = r_dict["question"]
            options = r_dict["options"]
            answer = r_dict["answer"]
            counterpart_idx = r_dict["counterpart_idx"]
            image_list = r_dict["image_list"]
            image_path_list = []
            for image_sample in image_list:
                image_name = image_save_path + image_sample["path"]
                if not os.path.exists(image_name):
                    image_byte = image_sample["bytes"]
                    image = Image.open(io.BytesIO(image_byte))
                    image.save(image_name, "JPEG")
                assert os.path.exists(image_name)
                image_path_list.append(image_name)
            datasample = {
                "sample_id":sample_id,
                "image_names":image_path_list,
                "question":question,
                "answer":answer,
                "options":options.tolist(),
                "task_name":task_name,
                "image_type":image_type,
                "image_relation":image_relation,
                "counterpart_idx":counterpart_idx
            }
            # print(datasample)
            fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
            fs_paths.flush()
    
    fs_paths.close()
    print("finished")
    