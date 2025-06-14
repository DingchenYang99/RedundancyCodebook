import os
import json

def load_data_for_flickr30k_test(flickr_path):
    anno_file = flickr_path + 'flickr30k_test-31kt-local.jsonl'
    img_path = flickr_path + 'images/'
    data = {'test': []}
    # flickr_anno = json.load(open(anno_file))
    flickr_anno = (json.loads(q) for q in open(anno_file, 'r', encoding='utf-8'))
    cnt = 0
    for anno in flickr_anno:
        sample_id = anno["sample_id"]
        gt_captions = anno["caption"]
        domain = 'test'
        file_name = anno["image_name"]
        data[domain] += [{
            "image_id": sample_id,
            "file_name": file_name,
            "gt_caption": gt_captions,
            "domain": domain
        }]
        cnt += 1
    print(f"number of samples: {cnt}")
    return data
        