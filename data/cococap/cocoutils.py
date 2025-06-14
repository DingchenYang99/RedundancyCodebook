import json

def load_data_for_inference(annot_path):
    annotations = json.load(open(annot_path))['images']

    data = {'test': [], 'val': []}

    for item in annotations:
        if item['split'] == 'train':
            continue
        file_name = item['filename'].split('_')[-1]
        all_captions = []
        for sentence in item['sentences']:
            this_caption = ' '.join(sentence['tokens'])
            all_captions.append(this_caption)
        image = {'file_name': file_name, 
                 'image_id': str(item['cocoid']),
                 'gt_caption': all_captions}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)
    print(f"number of val samples: {len(data['val'])}")
    print(f"number of test samples: {len(data['test'])}")
    return data

def load_data_for_token_reduc_exp(annot_path):
    annotations = json.load(open(annot_path))['images']

    data = {'train': [], 'test': [], 'val': []}

    for item in annotations:
        
        file_name = item['filename'].split('_')[-1]
        all_captions = []
        for sentence in item['sentences']:
            this_caption = ' '.join(sentence['tokens'])
            all_captions.append(this_caption)
        image = {'file_name': file_name, 
                 'image_id': str(item['cocoid']),
                 'gt_caption': all_captions}
        
        if item['split'] == 'train':
            data['train'].append(image)
        elif item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)
        else:
            pass
        
    print(f"number of train samples: {len(data['train'])}")
    print(f"number of val samples: {len(data['val'])}")
    print(f"number of test samples: {len(data['test'])}")
    return data

def load_data_for_try(annot_path):
    annotations = json.load(open(annot_path))['images']
    desired_image_ids = [
        # "000000038726.jpg", "000000278078.jpg", "000000371871.jpg", "000000397291.jpg",
        # "000000496402.jpg", "000000121110.jpg", "000000036149.jpg", "000000401969.jpg"
        # "000000298788.jpg", "000000079322.jpg", "000000088669.jpg", "000000023440.jpg"
        # "000000047512.jpg", "000000565160.jpg", "000000071857.jpg", "000000288986.jpg"
        # "000000532194.jpg", "000000548390.jpg", "000000074788.jpg", "000000041597.jpg"
        # "000000343821.jpg", "000000509691.jpg", "000000503561.jpg", "000000152458.jpg"
        "000000243173.jpg", "000000552520.jpg", "000000195317.jpg", "000000265597.jpg"
    ]
    # data = {'test': [], 'val': []}
    restval_num = 0
    train_num = 0
    val_num = 0
    test_num = 0
    all_num = 0
    for item in annotations:
        # print(str(item['cocoid'])+".jpg")
        if item['filename'].split('_')[-1] in desired_image_ids:
            print(item['filename'].split('_')[-1], item['split'])
        all_num += 1
        if item['split'] == 'train':
            train_num += 1
        elif item['split'] == 'restval':
            restval_num += 1
        elif item['split'] == 'test':
            test_num += 1
        elif item['split'] == 'val':
            val_num += 1
    print(f"number of val samples: {val_num}")  # 5000
    print(f"number of test samples: {test_num}")  # 5000
    print(f"number of restval samples: {restval_num}")  # 30504
    print(f"number of train samples: {train_num}")  # 82783
    print(f'sum: {val_num + test_num + restval_num + train_num}')
    print(f"number of all samples: {all_num}")  # 123287
    
if __name__ == '__main__':
    coco_anno_path = '/home/lufan/Projects/smallcap/caption/annotations/dataset_coco.json'
    load_data_for_try(coco_anno_path)