import os
import pandas as pd
import json
import re
from tqdm import tqdm
from PIL import Image
import io
from pathlib import Path

if __name__ == '__main__':
    
    source_path = '/DATA2/yangdingchen/flickr30k/'
    
    file_path = source_path + 'data/'
    image_root = source_path + 'images/'
    file_names = [fi for fi in sorted(os.listdir(file_path)) if fi.endswith(".parquet")]
    
    paths = source_path + 'flickr30k_test-31kt-local.jsonl'  # TODO
    print(paths)
    if os.path.exists(paths):
        print(paths+" already exists")
        raise ValueError
    
    fs_paths = open(paths, 'w', encoding="utf-8")
    cnt = 0
    for file_name in tqdm(file_names):
        
        df = pd.read_parquet(os.path.join(file_path, file_name))
        this_parq_id = file_name.split("-")[1]
        
        for r_idx, r in tqdm(df.iterrows()):
            r_dict = r.to_dict()
            img_id = r_dict["img_id"]
            image_id = r_dict["filename"]
            sent_ids = r_dict["sentids"]
            caption = r_dict["caption"]
            
            sample_id = f'flickr30k_test_{this_parq_id}_{img_id}_{image_id}'
            raw_image = r_dict["image"]
            
            # try:
            image_byte = raw_image["bytes"]
            image_filename = raw_image["path"]
            assert image_filename == image_id
            assert image_filename.split(".")[-1] in ["jpg", "jpeg", "png", "webp"]
            
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
                "img_id":img_id,
                "caption":caption.tolist(),
                "sent_ids":sent_ids.tolist(),
            }
            # print(datasample)
            cnt += 1
            fs_paths.write(json.dumps(datasample, ensure_ascii=False)+"\n")
            fs_paths.flush()
            
            # if cnt == 10:  # TODO
            #     break
            
    fs_paths.close()
    print(f"finished loading {cnt} valid samples")