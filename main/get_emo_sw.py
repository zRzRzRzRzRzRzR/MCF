import copy
import os
import json
import argparse
import re
from collections import defaultdict

from mpmath import floor
from tqdm import tqdm
import concurrent.futures
from utils import call_large_model, parse_json_response, load_yaml_config, merge_similar_emotions_with_llm


def extract_speaker_timestamps(txt_file_path):
    """
    从txt文件中提取说话人和时间戳
    格式：
        说话人 timestamp
        说话内容
        说话人 timestamp
        说话内容
        ...

    返回: 字典 {speaker: [timestamp1, timestamp2, ...]}
    """
    speaker_timestamps = defaultdict(list)
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i+1 >= len(lines):
                break
            line = lines[i].strip()
            parts = line.split()
            if len(parts) < 2:
                continue
            timestamp = parts[-1]
            speaker = ' '.join(parts[:-1])
            speaker_timestamps[speaker].append(timestamp)
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return {}
    return speaker_timestamps


def format_chat_history_for_llm(chat_data):
    formatted_chat = []

    for idx, item in enumerate(chat_data):
        holder = item.get("holder", "")
        sentence = item.get("input_sentence", "")
        tuples = item.get("final_model_response", [])
        formatted_tuples = []
        try:
            for t in tuples:
                formatted_tuples.append(
                    f'{{"target": "{t["target"]}", "aspect": "{t["aspect"]}", '
                    f'"opinion": "{t["opinion"]}", "sentiment": "{t["sentiment"]}", "rationale": "{t["rationale"]}"}}'
                )
            tuples_str = "[\n  " + ",\n  ".join(formatted_tuples) + "\n]" if formatted_tuples else "[]"
        except:
            tuples_str = "[]"
        if holder and sentence:
            formatted_chat.append(f'({idx}) "{holder}": "{sentence}"\n五元组: {tuples_str}')

    return "[\n" + ",\n".join(formatted_chat) + "\n]"


def segment_events_by_topic_with_sliding_window(dialogues, api_key, base_url, model_name, window_size=10, step_size=8, speaker_timestamps=None,other_text=None):
    total_sentences = len(dialogues)
    print(f"共计{total_sentences}个句子，切分成{floor(total_sentences // step_size) + 1}个窗口进行滑动")
    event_pool = defaultdict(lambda: {"events": []})
    step_results = []
    for start in range(0, total_sentences, step_size):
        print(f"now start at {start} with {step_size}, all is {total_sentences}")
        window_data = dialogues[start : start + window_size]
        formatted_chat = format_chat_history_for_llm(window_data)

        event_pool_copy = copy.deepcopy(event_pool)
        for holder_data in event_pool_copy.values():
            for event in holder_data["events"]:
                event.pop("sentence_ids", None)

        history_formatted = json.dumps(event_pool_copy, ensure_ascii=False, indent=2)
        
        # 格式化说话人和时间戳信息
        speaker_timestamps_json = json.dumps(speaker_timestamps, ensure_ascii=False, indent=2) if speaker_timestamps else "{}"
        
        system_prompt = """
你是一名高级情绪事件分析助手。你的任务是：
1. **分析对话数据**，识别出 **关键情绪事件**（event）。
2. **合并相似事件**，避免重复创建多个 event。
3. **保证情绪演变的连贯性**，记录关键的情绪变化，而不是简单重复相同情绪。

### 数据特点 
1. 这是段对话历史包含多个说话人，语言非常口语化。由于我给你提供的窗口是滑动的二不是完整的历史记录，可能当前句子属于上一个句子的event，而仅仅是一个reason。这种时候你需要合并在一个历史事件中，而不是创建一个新的事件。
2. 可能存在某个角色连续多次发言的情况。
3.你需要结合其他的输入进行分析，这些输入将会是文本模态的
4. 当前人说话可能没有任何情绪或参与到任何事件，比如主持人活着旁白，这种时候，events允许是空的。

### 输入数据中历史记录和文本输入的结构是：
历史记录:
[
    {"holder": "说话人1", "sentence": "说话内容1"}
    {"holder": "说话人2", "sentence": "说话内容2"}
    {"holder": "说话人1", "sentence": "说话内容3"}
    ...（多条记录）
]
其他输入：
{
  "音频文件信息": {
    "文件路径": "/root/autodl-fs/9.12.MP3",
    "音频时长": "357.49秒",
    "采样率": "16000Hz"
  },

  "音频概览": {
    "总时长": "10分钟",
    "说话人数量": 2,
    "音频质量": "一般"
  },

  "1_voice": {
    "声音特征": "男性，年龄在30岁左右，音色低沉",
    "语速": "平均语速，没有明显加速或减速",
    "语调": "基本语调为中性，没有明显的情感表达",
    "音高": "基础音高水平，音高变化幅度较小",
    "情绪": "中性情绪，没有明显的情感波动",
    "内容": "主要讲述一些日常生活琐事"
  },
  "2_voice": {
    "声音特征": "女性，年龄在25岁左右，音色清脆",
    "语速": "语速适中，没有明显的加速或减速",
    "语调": "基本语调为升调，情感表达较为积极",
    "音高": "基础音水平较高，音高变化幅度较大",
    "情绪": "愉快的情绪，表现出较高的幸福感",
    "内容": "主要与生活琐事和朋友间的互动有关"
  },
  "3_voice": {
    "声音特征": "女性，年龄在28岁左右，音色柔和",
    "语速": "语速较快，有时会加速",
    "语调": "基本语调为升调，有波动的情感表达",
    "音高": "基础音水平高，音高变化幅度较大",
    "情绪": "愉快的情绪，表现出较高的幸福感",
    "内容": "主要与生活琐事，和朋友间的互动较为友好，没有明显的冲突或争执，对话内容涉及日常生活琐事和情感交流。"
  },

  "交互特征分析": {
    "话轮模式": "说话人3先发言，说话人2随后发言，交替进行，说话人1进行补充",
    "互动特点": "三位说话人之间的互动较为友好，没有明显的冲突或争执，对话内容涉及日常生活琐事和情感交流。"
  },
  "1": {
    "emotions": {
      "00:01": "happy",
      "00:19": "sad",
      "01:03": "angry",
      "02:14": "surprised",
      "02:31": "neutral",
      "03:05": "happy"
      ...(多条记录)
}
}

### 输出格式要求

1. JSON 格式，外层结构应只包含当前角色的 ID，例如 `"1": { "events": [...] }`。
2. `events` 为一个列表，每个事件包含：
    - "event"（事件名称，通常是一个很大的事件或者一个场景，每个角色的两个话题差异通常是很大的）
    - "emotions"（角色在该事件中的情绪列表和每个阶段导致这个情绪的原因）

3. `emotions` 结构：
    - 只记录**关键的情绪变化**，避免相同情绪重复
    - `state` 选项：`["positive", "negative", "neutral", "ambiguous", "doubt"]`
    - `reason` 需要描述角色情绪变化的依据或原因。这通常是这个事件的进展，或者更换了话题，比如实验中的某个话题。
    
具体格式如下：
```json
{
    “角色ID_1”: {
    “events”: [
                {
                “event”: 事件名称,
                “sentence_ids”: 跟这个事件相关的句子编号，是一个列表，返回()中的数字，比如第一句，第二句，第五句，返回[1,2,5]
                “emotions”: [
                                {
                                    “source_id”: 角色ID(只有一个, 一个数字)，如果这个情感变化是由于某一个角色（一定是角色ID）导致，则输出对应角色ID。如果没有指定，或者由于上下文理解为自己的情感变化，则为自己的角色ID。这个句子一定出现在所有的holders内。
                                    “state”: “positive/negative/neutral/ambiguous/doubt”,
                                    “reason”: 该事件产生该情绪的原因
                                }
                            ]
                        }
                    ]
                },
    “角色ID_2”: {
                … 相同的结构
                }
}



这是一个返回的例子:

```json
{
    “1”: {
            “events”: [
                        {
                        “event”: “朋友拒绝分享食物”,
                        “sentence_ids”: [0, 2, 5]
                        “emotions”: [
                                        {
                                            “source_id”: 3,
                                            “state”: “negative”,
                                            “reason”: “对方不愿意分享食物，让自己感到不开心”
                                        },
                                        {
                                    ]
                                }
                            ]
                        },
    “2”: {
            “events”: [
                        {
                        “event”: “朋友拒绝分享食物”,
                        “sentence_ids”: [0, 2, 5]
                        “emotions”: [
                                        {
                                            “source_id”: 3,
                                            “state”: “negative”,
                                        }
                        ]
                        }
            ]
     }
}

“2”: {
“events”: []
}
}


在实际的场景中，"source_id"要根据上下文判断，3 和 2 仅仅是假设，实际句子中，角色ID(只有一个, 一个数字)，如果这个情感变化是由于某一个角色（一定是角色ID）导致，则输出对应角色ID。如果没有指定，或者由于上下文理解为自己的情感变化，则为自己的角色ID。这个句子一定出现在所有的holders内。

### 注意

- 不要遗漏任何关键字段，source_id 一定要在emotions中输出，这个字段不能省略。
- 同一事件中若情绪未发生变化，不应重复记录。必须根据事件主题，合并情绪变化的原因。
- 每个角色都必须有输出，如果没有明显的事件，需要输出空列表。这很重要。 同一个人在一个数据集中通常不会有太多event。你应该给出尽量少的event事件，具体的情感变化应该放在emotions字段中。
- event, reason 都应该简单，表达清晰。emotions中可以列出详细的情感变化。
- sentence_ids 不要输出句子，只要对应的()内的句子ID。没有"sentences"字段，不允许输出任何原始的句子!,只能输出"sentences_id"字段，这是针对这个事件的句子ID。
- 不需要带有任何解释,只能返回要求内的内容。输出格式必须完全和 ### 输出格式要求的一样。 返回的必须是一个dict而不是一个list结构的json，最外层必须是dict。
    
请直接返回 JSON，不能有多余解释。
""".strip()

        user_prompt = f"""
[之前已经检测到的事件]
{history_formatted}
[相关历史记录]
{formatted_chat}
[说话人时间戳摘要]
{speaker_timestamps_json}
[其他文本输入]
{other_text}
- **请按照格式输出 JSON**，不要遗漏任何关键字段，source_id 一定要在emotions中输出，这个字段不能省略。
        """
        parsed_response = None
        for _ in range(3):
            response = call_large_model(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                api_key=api_key,
                base_url=base_url,
                model=model_name,
            )
            parsed_response = parse_json_response(response)
            if parsed_response and parsed_response != {}:
                break
        if not parsed_response:
            parsed_response = {}
        for holder, holder_data in parsed_response.items():
            if holder not in event_pool:
                event_pool[holder] = {"events": []}
            try:
                for new_event in holder_data["events"]:
                    new_event["sentence_ids"] = [start + idx for idx in new_event.get("sentence_ids", [])]
                    existing_event = next(
                        (e for e in event_pool[holder]["events"] if e["event"] == new_event["event"]), None
                    )
                    if existing_event:
                        existing_event["emotions"].extend(new_event["emotions"])
                        existing_event["sentence_ids"].extend(new_event["sentence_ids"])
                    else:
                        event_pool[holder]["events"].append(new_event)
            except Exception as e:
                print(f"Error processing {holder} at {start} with {step_size}")
        print("enter in event_pool\n=======================\n")
        for holder_data in event_pool.values():
            for event in holder_data["events"]:
                event["sentence_ids"] = list(set(event["sentence_ids"]))
                event["sentences"] = [
                    dialogues[idx]["input_sentence"] for idx in event["sentence_ids"] if idx < len(dialogues)
                ]
        step_results.append({"step": start // step_size + 1, "events": parsed_response})
    for holder, holder_data in event_pool.items():
        for event in holder_data["events"]:
            optimized_emotions = merge_similar_emotions_with_llm(event["emotions"], api_key, base_url, model_name)
            event["emotions"] = optimized_emotions

    return event_pool, step_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--other_text",type=str,required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--window_sizes", type=str, default="20", help="滑动窗口大小，多个用逗号分隔，比如 '10,40'")
    parser.add_argument("--step_sizes", type=str, default="10", help="滑动步长，多个用逗号分隔，比如 '8,30'")
    args = parser.parse_args()
    window_sizes = list(map(int, args.window_sizes.split(",")))
    step_sizes = list(map(int, args.step_sizes.split(",")))
    other_text = args.other_text

    if len(window_sizes) != len(step_sizes):
        raise ValueError("需要提供相同数量的滑动窗口和步长组合")

    llm_cfg = load_yaml_config(args.config_path, args.llm_model, "llm_config")
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json") and f.startswith("output_chat")])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as executor:
        with tqdm(total=len(all_files) * len(window_sizes) * len(step_sizes), desc="Processing files") as pbar:
            futures = {}

            for fname in all_files:
                file_path = os.path.join(args.input_dir, fname)
                # 尝试读取对应的txt文件
                txt_file_path = os.path.join(args.input_dir, fname.replace('.json', '.txt'))
                speaker_timestamps = extract_speaker_timestamps(txt_file_path)
                other_text = args.other_text
                with open(file_path, "r", encoding="utf-8") as f:
                    dialogues = json.load(f)

                for window_size, step_size in zip(window_sizes, step_sizes):
                    match = re.search(r"(\d+)", fname)
                    if match:
                        number = match.group(1)
                    else:
                        number = "unknown"
                    event_output_path = os.path.join(args.output_dir, f"output_emotions_{number}_events.json")
                    step_output_path = os.path.join(args.output_dir, f"other_text_{number}.json")
                    if os.path.exists(event_output_path) and os.path.exists(step_output_path):
                        pbar.update(1)
                        print(f"Skipping {fname} with window_size={window_size}, step_size={step_size}")
                        continue

                    futures[
                        executor.submit(
                            segment_events_by_topic_with_sliding_window,
                            dialogues,
                            llm_cfg["api_key"],
                            llm_cfg["base_url"],
                            llm_cfg["model"],
                            window_size,
                            step_size,
                            speaker_timestamps,  # 传递说话人和时间戳信息
                            other_text
                        )
                    ] = (fname, window_size, step_size)

            # Handle task results as they complete
            for future in concurrent.futures.as_completed(futures):
                fname, window_size, step_size = futures[future]
                event_pool, step_results = future.result()

                match = re.search(r"(\d+)", fname)
                if match:
                    number = match.group(1)
                else:
                    number = "unknown"

                output_file_path = os.path.join(args.output_dir, f"output_emotions_{number}_events.json")
                step_results_path = os.path.join(args.output_dir, f"output_emotions_{number}_steps.json")
                print(f"{output_file_path} is saved.")
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(event_pool, f, ensure_ascii=False, indent=2)

                with open(step_results_path, "w", encoding="utf-8") as f:
                    json.dump(step_results, f, ensure_ascii=False, indent=2)

                pbar.update(1)


if __name__ == "__main__":
    main()
