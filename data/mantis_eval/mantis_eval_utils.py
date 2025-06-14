import os
import json

def get_ques_templates(templates_path):
    all_templates = {}
    for template_file in templates_path.glob('*.txt'):
        with open(template_file, 'r') as f:
            all_templates[template_file.stem] = f.read()
    return all_templates

def load_data_for_mantis_eval(mantis_path,mantis_ques_temp_path):
    anno_file = mantis_path + 'mantis_eval_test_217t-local.jsonl'
    img_path = mantis_path + 'images/'
    data = {'test': []}
    mantis_anno = [json.loads(q) for q in open(anno_file, 'r', encoding='utf-8')]
    cnt = 0
    all_templates = get_ques_templates(mantis_ques_temp_path)
    for anno in mantis_anno:
        sample_id = anno["sample_id"]
        image_names = anno["image_names"]
        all_img_valid_flag = True
        for image_name in image_names:
            if not os.path.exists(image_name):
                all_img_valid_flag = False
        if not all_img_valid_flag:
            print(f"{sample_id} has invalid image, skipped, plz check!")
            continue
        question = anno["question"]
        options = anno["options"]
        answer = anno["answer"]
        category = anno["category"]
        question_type = anno["question_type"]
        raw_ques = question
        if question_type == 'multi-choice':
            option_idx = 'A'
            for option in options:
                if not any([x in option.upper() for x in [f"{option_idx})", f"{option_idx}:", f"{option_idx}."]]):
                    question += f'\n ({option_idx}) {option}'
                else:
                    question += f'\n {option}'
                option_idx = chr(ord(option_idx) + 1)
        template = all_templates[question_type]
        question = template.format(question=question)
        
        data['test'] += [{
            "sample_id": sample_id,
            "image_names": image_names,
            "question":question,
            "answer": answer,
            "raw_question":raw_ques,
            "options":options,
            "question_type":question_type,
            "category":category,
        }]
        cnt += 1
    print(f"number of samples: {cnt}")
    return data