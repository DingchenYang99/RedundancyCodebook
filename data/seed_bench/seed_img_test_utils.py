import os
import json

def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."

def load_data_for_seed_img_test(seed_path):
    anno_file = seed_path + 'seed_bench_image-17kt-local.jsonl'
    img_path = seed_path + 'images/'
    data = {'image': []}
    seed_anno = [json.loads(q) for q in open(anno_file, 'r', encoding='utf-8')]
    cnt = 0
    for anno in seed_anno:
        data_type = anno["data_type"]
        if data_type != 'image':
            continue
        sample_id = anno["sample_id"]
        answer = anno["answer"]
        choice_a = anno["choice_a"]
        choice_b = anno["choice_b"]
        choice_c = anno["choice_c"]
        choice_d = anno["choice_d"]
        question = anno["question"]
        image_name = anno["image_name"]
        if not os.path.exists(image_name):
            print(f"{sample_id} has invalid image, skipped, plz check!")
            continue
        question_id = anno["question_id"]
        data_id = anno["data_id"]
        question_type_id = anno["question_type_id"]
        
        raw_ques = question
        question_with_option_instruct = seed_doc_to_text(anno)
        
        data['image'] += [{
            "sample_id":sample_id,
            "image_name":image_name,
            "question":question_with_option_instruct,
            "answer":answer,
            "raw_question":raw_ques,
            "choice_a":choice_a,
            "choice_b":choice_b,
            "choice_c":choice_c,
            "choice_d":choice_d,
            "data_type":data_type,
            "data_id":data_id,
            "question_id":question_id,
            "question_type_id":question_type_id,
        }]
        cnt += 1
    print(f"number of samples: {cnt}")
    return data