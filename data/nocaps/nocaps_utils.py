import os
import json

def load_data_for_nocaps(nocaps_path):
    nocaps_anno_file = nocaps_path + 'nocaps_val_4500_captions.json'
    nocaps_img_path = nocaps_path + 'images/'
    data = {'in-domain': [], 'out-domain': [], "near-domain": []}
    nocaps_anno = json.load(open(nocaps_anno_file))
    nocaps_anno_images = nocaps_anno['images']
    nocaps_anno_annotation = nocaps_anno['annotations']
    annotation_dict = {}
    for anno in nocaps_anno_annotation:
        nocaps_image_id = anno["image_id"]
        if nocaps_image_id not in annotation_dict.keys():
            annotation_dict[nocaps_image_id] = []
            annotation_dict[nocaps_image_id] += [anno["caption"]]
        else:
            annotation_dict[nocaps_image_id] += [anno["caption"]]
    for img_dict in nocaps_anno_images:
        nocaps_image_id = img_dict["id"]
        domain = img_dict["domain"]
        open_images_id = img_dict["open_images_id"]
        file_name = img_dict["file_name"]
        assert domain in data.keys()
        gt_captions = annotation_dict[nocaps_image_id]
        data[domain] += [{
            "image_id": open_images_id,
            "file_name": file_name,
            "gt_caption": gt_captions,
            "domain": domain
        }]
    print(f"number of in-domain samples: {len(data['in-domain'])}")
    print(f"number of near-domain samples: {len(data['near-domain'])}")
    print(f"number of out-domain samples: {len(data['out-domain'])}")
    return data
        