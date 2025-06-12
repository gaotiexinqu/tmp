"""
支持N轮
"""

import json
import random
import re

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def match(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        raw = match.group(1)
        try:
            tokens = raw.replace(",", " ").split()
            answer_list = [int(x) for x in tokens]
        except Exception as e:
            answer_list = []
    else:
        answer_list = []
    return answer_list

if __name__ == "__main__":
    correct, total = 0, 0
    num_epoch = 1

    data_w = []
    input_file = r'G:\Temporal-RL\data\eval\v1_test117_Temporal-RL-v1-SFT-Model.jsonl'
    
    data = read_jsonl(input_file)
    for item in data:
        response = item['response']
        if type(response) == str:
            answer_list = match(response)
            if answer_list != []:
                total += 1
            if answer_list == item["gt_order"]:
                data_w.append(item)
                print(item["video_path"])
                correct += 1
        elif type(response) == list:
            total = len(data)

            for idx, resp in enumerate(response):
                if idx >= num_epoch:
                    break
                
                answer_list = match(resp)
                if answer_list == item["gt_order"]:
                    correct += 1
                    break

    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print(f"原始数据量：{len(data)}， 筛选后数据量：{len(data_w)}")
    print("saved!")

    # decide if save for reject sampling(GPT)
    # save_jsonl(data_w, r'G:\Temporal-RL\data\GPT4o_sum_caption_shuffle-order_processed_gpt4o_correct.jsonl')