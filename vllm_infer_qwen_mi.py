import json
import os
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


prompt_system = """
You are an experienced expert in video frame reordering, skilled at restoring the correct temporal order of video frames based on semantic clues.

# The user will provide the following:
- {num_frames} images, which are keyframes from the same video clip but have been shuffled out of order;
- A caption that describes the video in natural language, outlining the temporal progression of the events depicted across the {num_frames} frames.

To distinguish between the images, the user has assigned each image a numerical index from left to right: 1, 2, 3, ..., meaning the N-th image in the input is indexed as N.

# Your task:
Analyze the content of the {num_frames} images in relation to the caption and reorder them to match the temporal sequence described in the caption.

For example, if the caption says:
“A person first picks up an egg, then puts it down, and finally picks up a spatula,”
then you should:
Identify the frame showing the person picking up the egg and place it earlier in the sequence;
Then include the frame of the person putting the egg down;
Finally, place the frame where the person picks up the spatula.
Disregard the original file names or image indices—only use the visual content of the images and the caption to determine the correct order.

# Important guidelines:
🚫 Do not rely on the filenames or original numbering of the images;
✅ Base the ordering entirely on the sequence of events described in the caption;
🧠 Clearly explain your reasoning process, justifying why each image is placed in a specific position;
💡 The final order should reflect a coherent and logical progression of events;
🔢 The output should be the original indices of the images (as provided by the user), enclosed within <answer> and </answer> tags.

# Example output format:
<answer>3, 1, 2</answer>
"""

prompt_user = """
Below is the user-provided caption describing the video:
{caption}
"""


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# data
media_root = "/home/bingxing2/ailab/quyichang/zhaoy/data/v3/frames_key"
meta_path = "/home/bingxing2/ailab/quyichang/zhaoy/code/vllm_infer/data/test_v2_200.jsonl"

model_path = "/home/bingxing2/ailab/quyichang/zengyu/model/Qwen2.5-VL-3B-Instruct"
# model_path = "/home/bingxing2/ailab/quyichang/zhaoy/models/Temporal-RL-v1-SFT-Model"
model_name = model_path.split("/")[-1]

save_path = f"/home/bingxing2/ailab/quyichang/zhaoy/code/vllm_infer/data/test_v2_200_{model_name}.jsonl"
data = read_jsonl(meta_path)
num_epoch = 10

llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 20, "video": 10},
)

# default
# sampling_params = SamplingParams(
#     temperature=0.1,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=256,
#     stop_token_ids=[],
# )

sampling_params = SamplingParams(
    temperature=0.8,  # rollout得设高点...
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.05,
    max_tokens=4096,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(model_path)

data_w = []
for item in tqdm(data):

    content = []
    for img in item["frame_path_shuffle"]:
         content.append({
                    "type": "image",
                    "image": os.path.join(media_root, img),
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                })
    content.append({"type": "text", "text": prompt_user.format(caption=item["sum_caption"])})
    messages = [
        {"role": "system", "content": prompt_system.format(num_frames=len(item["frame_path_shuffle"]))},
        {
            "role": "user",
            "content": content
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,

        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    item["response"] = []
    for i in range(num_epoch):
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        item["response"].append(generated_text)
    data_w.append(item)

save_jsonl(data_w, save_path)