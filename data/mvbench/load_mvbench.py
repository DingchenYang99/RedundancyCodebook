import os
import json
import sys
sys.path.append("/home/lufan/Projects/PensieveV2/experiments/data/mvbench/")

from mvbench_utils import mvbench_doc_to_text, mvbench_doc_to_visual, mvbench_frames_doc_to_visual

data_list = {
    "Action Sequence": ("action_sequence.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "your_data_path/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "pyour_data_path/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "your_data_path/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "your_data_path/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "your_data_path/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "your_data_path/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "your_data_path/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "your_data_path/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "your_data_path/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "your_data_path/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "your_data_path/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "your_data_path/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "your_data_path/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "your_data_path/clevrer/video_validation/", "video", False),
}

name_map_ = {
    "object_interaction": "Object Interaction",
    "action_sequence": "Action Sequence",
    "action_prediction": "Action Prediction",
    "action_localization": "Action Localization",
    "moving_count": "Moving Count",
    "fine_grained_pose": "Fine-grained Pose",
    "character_order": "Character Order",
    "object_shuffle": "Object Shuffle",
    "egocentric_navigation": "Egocentric Navigation",
    "moving_direction": "Moving Direction",
    "episodic_reasoning": "Episodic Reasoning",
    "fine_grained_action": "Fine-grained Action",
    "scene_transition": "Scene Transition",
    "state_change": "State Change",
    "moving_attribute": "Moving Attribute",
    "action_antonym": "Action Antonym",
    "unexpected_action": "Unexpected Action",
    "counterfactual_inference": "Counterfactual Inference",
    "object_existence": "Object Existence",
    "action_count": "Action Count",
}

def load_data_for_mvbench_test(mvbench_path, interested_tasks):
    print(f"loading mvbench data...")
    data_dir_json = os.path.join(mvbench_path, "json")
    data_dir_video = os.path.join(mvbench_path, "video")
    data = {}
    total_cnt = 0
    for interested_task in interested_tasks:
        if interested_task not in name_map_.keys():
            print(f"{interested_task} is unknown plz check!")
            raise KeyError
        if interested_task not in data:
            data[interested_task] = []
            
        meta_data = data_list[name_map_[interested_task]]
        json_name = meta_data[0]
        read_modality = meta_data[2]
        use_bound = meta_data[3]
        this_anno = os.path.join(data_dir_json, f"json_{json_name}")
        
        if not os.path.exists(this_anno):
            print(f"{this_anno} does not exist, plz check!")
            raise ValueError
        
        this_fr = json.load(open(this_anno, 'r', encoding='utf-8'))
        assert len(this_fr) >= 1
        
        cnt = 0
        for anno in this_fr:
            video_name = anno["video"]
            video_id = video_name.split(".mp4")[0] if ".mp4" in video_name else video_name
            sample_id = interested_task + '-' + video_id
            
            if read_modality == 'video':
                video_path_list = mvbench_doc_to_visual(anno, data_dir_video, interested_task)
            elif read_modality == 'frame':
                video_path_list = mvbench_frames_doc_to_visual(anno, data_dir_video, interested_task)
            
            all_img_valid_flag = True
            for image_name in video_path_list:
                if not os.path.exists(image_name):
                    all_img_valid_flag = False
            if not all_img_valid_flag:
                print(f"{sample_id} has invalid ƒrame or video, skipped, plz check!")
                continue
            if "fps" in anno:
                fps = anno["fps"]
            else:
                fps = None
            question = anno["question"]
            candidates = anno["candidates"]
            answer = anno["answer"]
            if use_bound:
                start = float(anno["start"])
                end = float(anno["end"])
                bound = [start, end]
            else:
                bound = None
            
            answer_option = None
            for idx, c in enumerate(candidates):
                if c == answer:
                    answer_idx = idx
                    answer_option = chr(ord('A') + answer_idx)
                    break
            # print(answer_option)
            question_option = mvbench_doc_to_text(anno).strip()
            
            data[interested_task] += [{
                "sample_id": sample_id,
                "video_path_list": video_path_list,
                "question":question_option,
                "answer": answer,
                "answer_option":answer_option,
                "options":candidates,
                "raw_question":question,
                "task_name":interested_task,
                "read_modality":read_modality,
                "use_bound":use_bound,
                "bound":bound,
                "fps":fps,
                "new_task_name":name_map_[interested_task]
            }]
            cnt += 1
        print(f"number of task {interested_task} samples: {cnt}")
        total_cnt += cnt
    print(f"successfully loaded all {total_cnt} samples for mvbench")
    return data