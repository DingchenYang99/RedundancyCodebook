import os
import json
import sys
import pandas as pd

# all copied from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmbench/

def create_options_prompt(row_data, option_candidate):
        sys_prompt = "There are several options:"
        available_keys = set(row_data.keys()) & set(option_candidate)
        options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
        sorted_options = dict(sorted(options.items()))
        options_prompt = f"{sys_prompt}\n"
        for key, item in sorted_options.items():
            if pd.notna(item) and item != "nan":
                options_prompt += f"{key}. {item}\n"
        return options_prompt.rstrip("\n"), sorted_options
    
lmms_eval_specific_kwargs = {
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    }

def mmbench_doc_to_text(doc):
    
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = create_options_prompt(doc, option_candidate)

    data = {
        "sample_id": doc["sample_id"],
        "image_name": doc["image_name"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "l2-category": doc["l2_category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']} {data['question']} {data['options']}" \
        if pd.notna(data["hint"]) and data["hint"] != "nan" else f"{data['question']} {data['options']}"

    # if lmms_eval_specific_kwargs:
    query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt, data

def load_data_for_mmb_en_dev(mmb_path):
    anno_file = mmb_path + 'mmb_en-11kt-local.jsonl'
    # img_path = mmb_path + 'images/'
    domain = 'mmb_en_dev'
    data = {domain: []}
    mmb_anno = [json.loads(q) for q in open(anno_file, 'r', encoding='utf-8')]
    cnt = 0
    for anno in mmb_anno:
        sample_id = anno["sample_id"]
        if not sample_id.startswith(domain):
            continue
        # index = anno["index"]
        image_name = anno["image_name"]
        # question = anno["question"]
        # answer = anno["answer"]
        # hint = anno["hint"]
        ans_A = anno["ans_A"]
        ans_B = anno["ans_B"]
        ans_C = anno["ans_C"]
        ans_D = anno["ans_D"]
        anno.update({"A":ans_A})
        anno.update({"B":ans_B})
        anno.update({"C":ans_C})
        anno.update({"D":ans_D})
        # category = anno["category"]
        # l2_category = anno["l2_category"]
        # source = anno["source"]
        # split = anno["split"]
        if not os.path.exists(image_name):
            print(f"{sample_id} has invalid image, skipped, plz check!")
            continue
        
        query_prompt, datasample = mmbench_doc_to_text(anno)
        datasample.update({"A":ans_A})
        datasample.update({"B":ans_B})
        datasample.update({"C":ans_C})
        datasample.update({"D":ans_D})
        
        datasample.update({"query_prompt":query_prompt})
        data[domain] += [datasample]
        cnt += 1
    print(f"number of valid samples: {cnt}")
    return data