import os
import json

lmms_eval_specific_kwargs = {
    "pre_prompt": "",
    "post_prompt": "\nAnswer with the option's letter from the given choices directly."
}

def muir_doc_to_text(doc):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def load_data_for_muir_test(muir_path):
    anno_file = muir_path + 'muir_bench_test_2kt-local.jsonl'
    # img_path = muir_path + 'images/'
    data = {'test': []}
    muir_anno = [json.loads(q) for q in open(anno_file, 'r', encoding='utf-8')]
    cnt = 0
    for anno in muir_anno:
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
        task_name = anno["task_name"]
        image_type = anno["image_type"]
        image_relation = anno["image_relation"]
        counterpart_idx = anno["counterpart_idx"]
        raw_ques = question
        question = muir_doc_to_text(anno)
        
        data['test'] += [{
            "sample_id": sample_id,
            "image_names": image_names,
            "question":question,
            "answer": answer,
            "options":options,
            "raw_question":raw_ques,
            "task_name":task_name,
            "image_type":image_type,
            "image_relation":image_relation,
            "counterpart_idx":counterpart_idx,
        }]
        cnt += 1
    print(f"number of MUIRBENCH test samples: {cnt}")
    return data
