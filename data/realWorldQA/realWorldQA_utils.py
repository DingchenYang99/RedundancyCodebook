import os
import json

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."

def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""  # TODO empty in https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/bb3fd824651336d6b001d86dd57a7042bf3bcf0b/lmms_eval/tasks/realworldqa/realworldqa.yaml for llava family
    post_prompt = ""  # TODO
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def load_data_for_realWorldQA_test(realworldQA_path):
    anno_file = realworldQA_path + 'realWorldQA_test-765t-local.jsonl'
    img_path = realworldQA_path + 'images/'
    data = {'test': []}
    rwqa_anno = [json.loads(q) for q in open(anno_file, 'r', encoding='utf-8')]
    cnt = 0
    for anno in rwqa_anno:
        sample_id = anno["sample_id"]
        question = anno["question"]
        answer = anno["answer"]
        domain = 'test'
        file_name = anno["image_name"]
        processed_question = realworldqa_doc_to_text(anno)
        data[domain] += [{
            "sample_id": sample_id,
            "file_name": file_name,
            "question": processed_question,
            "raw_question": question,
            "answer": answer,
        }]
        cnt += 1
    print(f"number of samples: {cnt}")
    return data
