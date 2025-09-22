import json
import re

import faiss
import numpy as np
import torch
import yaml
from openai import AzureOpenAI, OpenAI


def load_yaml_config(config_path, api_name, config_type="llm_config"):
    """
    从 YAML 配置文件中加载指定类型的 API 配置信息，并返回一个包含所有配置的字典。
    :param config_path: 配置文件路径
    :param api_name: API 的名称
    :param config_type: 配置类型（默认是 'llm_config'，也可以是 'embed_config'）
    :return: 包含模型配置的字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_type not in config:
        raise ValueError(f"配置文件中未找到 '{config_type}' 配置")

    api_config = config[config_type].get(api_name)
    if not api_config:
        raise ValueError(f"未找到 '{api_name}' 的配置")

    return {
        "model": api_config["model"],
        "base_url": api_config["base_url"],
        "api_key": api_config["api_key"],
    }


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def call_large_model(
    messages, api_key="EMPTY", base_url=None, model=None, version="2024-08-01-preview"
):
    if "azure" in base_url:
        client = AzureOpenAI(api_key=api_key, base_url=base_url, api_version=version)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=100000)
    try:
        for i in range(3):
            if "o3" in model or "o1" in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=16000,
                    stream=False,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8192 if "glm" in model else 16000,
                    stream=False,
                )
            if response.choices[0].message.content.strip():
                return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in call_large_model as {e}")
        return {}


def clean_response(response_str):
    response_str = response_str.strip()
    if response_str.startswith("```") and response_str.endswith("```"):
        lines = response_str.split("\n")
        lines = lines[1:-1]
        response_str = "\n".join(lines).strip()
    return response_str


def parse_json_response(response):
    response_str = clean_response(response)

    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass

    arr_match = re.search(r"\[[\s\S]*\]", response_str)
    if arr_match:
        arr_str = arr_match.group(0).strip()
        try:
            return json.loads(arr_str)
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", response_str)
    if obj_match:
        obj_str = obj_match.group(0).strip()
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            pass
    messages = [
        {
            "role": "system",
            "content": """
你将看到一个JSON输入 ，它包含一些格式错误。请帮助我修复它。这不是一个list，因此一定是单个json。
确保所有的键值对格式正确，并且所有括号和逗号都完整无误。返回修复后的有效JSON。

## 输出格式
```json
修复后的json

```

## 注意：

- 有时候，原始的输入包含不规则的符号或者连续超过100个数字，则应该把这个字段设为[]或者{}，因为这是输入错误。
- 不要输出额外内容。
- 所有输入的json冲重复的内容都需要删除。仅保留最简单的json，不要超过5000个token。

""",
        },
        {"role": "user", "content": f"原始JSON: \n{response_str}\n\n修复后的JSON:"},
    ]
    # FIXME： The incomplete JSON when calling the large model may result in the final output not conforming to the correct format. During validation, it may be necessary to manually add []
    fixed_json_str = call_large_model(
        messages, model="Qwen/Qwen2.5-72B-Instruct", base_url="http://localhost:8000/v1"
    )
    try:
        obj_match = re.search(r"```json\s*\n([\s\S]*?)\n```", fixed_json_str)
        if obj_match:
            obj_str = obj_match.group(0).strip().lstrip("```json\n").rstrip("```")
            return json.loads(obj_str)
    except json.JSONDecodeError:
        return [{}]


def remove_spaces(obj):
    if isinstance(obj, str):
        return obj.strip()
    elif isinstance(obj, list):
        return [remove_spaces(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: remove_spaces(v) for k, v in obj.items()}
    else:
        return obj


def call_embedding(texts, api_key, base_url, model="embedding-3"):
    if isinstance(texts, list):
        texts = [" " if text == "" else text for text in texts]

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=texts)
    embeddings = []
    for item in response.data:
        emb = np.array(item.embedding, dtype="float32")
        embeddings.append(emb)
    return embeddings


def call_embeddings_batch(texts, api_key, base_url, model="embedding-3", batch_size=10):
    client = OpenAI(api_key=api_key, base_url=base_url)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            emb = np.array(item.embedding, dtype="float32")
            embeddings.append(emb)
    return embeddings


def build_faiss_index(dimension):
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dimension)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    torch.cuda.empty_cache()
    return gpu_index


def merge_similar_emotions_with_llm(emotions, api_key, base_url, model_name):
    """
    使用大模型合并相似的情绪变化。
    :param emotions: 包含情绪变化的列表
    :param api_key: OpenAI API key
    :param base_url: OpenAI API base URL
    :param model_name: OpenAI 模型名称
    """
    if not emotions:
        return []

    system_prompt = """
你是一名高级情绪分析优化助手。你的任务是：
1. **合并重复的情绪变化**，如果 state 相同，则合并 reason，去掉重复信息。
2. **优化表达**，确保 reason 简洁但完整，避免冗余。
3. **不要更改原始事件的情绪 state**，只是优化表达。
4. **输出严格符合 JSON 结构**。

### **输入示例**
```json
[
    {"state": "negative", "reason": "对方不愿意分享食物"},
    {"state": "negative", "reason": "对方没有考虑到我的感受"},
    {"state": "doubt", "reason": "对方以前会吃这个，但今天却拒绝"}
]
```

### **优化后示例**
```json
[
    {"state": "negative", "reason": "对方不愿意分享食物，并没有考虑到我的感受"},
    {"state": "doubt", "reason": "对方以前会吃这个，但今天却拒绝"}
]
    ```
""".strip()

    user_prompt = f"""
请优化以下情绪变化：
{json.dumps(emotions, ensure_ascii=False, indent=2)}
"""

    response = call_large_model(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    optimized_emotions = parse_json_response(response)
    return optimized_emotions if optimized_emotions else emotions


def build_history_triples(results, current_index, history_num):
    start_idx = max(0, current_index - history_num)
    history_items = results[start_idx:current_index]
    history_list = []
    for idx, item in enumerate(history_items):
        quadruple = item.get("model_response", [])
        if not quadruple or not isinstance(quadruple, list):
            continue
        q = quadruple[0]

        aspect = q.get("aspect", "")
        target = q.get("target", "")
        sentiment = q.get("sentiment", "")
        rationale = q.get("rationale", "")

        opinions_val = q.get("opinion", q.get("opinions", q.get("ppinion", "")))

        history_list.append(
            {
                "sentence": item["input_sentence"],
                "holder": item["holder"],
                "target": target,
                "aspect": aspect,
                "opinion": opinions_val,
                "sentiment": sentiment,
                "rationale": rationale,
            }
        )
    return history_list
