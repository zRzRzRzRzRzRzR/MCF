# MCF: Text LLMS For Multimodal Emotional Causality

<div align="center">

[![arXiv](https://img.shields.io/badge/📚%20Arxiv-Coming%20soon-ff0000)](#)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-MCF-blueviolet)](https://huggingface.co/datasets/ZHANGYUXUAN-zR/MCF-Dataset)
</div>

## Data

Dataset task definition and annotation example of the MCF framework. The framework contains two core subtasks:
five-tuple
element extraction (identifying Target, Holder, Aspect, Opinion, Sentiment, and Rationale) and sentiment chain
analysis (constructing causal
relationship chains between emotional events).

![dataset.png](resources/dataset.png)

The dataset is provided in [MCF-Dataset](https://huggingface.co/datasets/ZHANGYUXUAN-zR/MCF-Dataset)
with the following structure.  
Each sample includes video, audio, and dialogue subtitles:

```
|-video
    |-chat_1.mp4

|-chat
    |-chat_1.txt

|-audio
    |-chat_1.mp3
```

## MCF pipeline

The MCF (Multimodal Causality Framework) architecture. MCF employs a three-stage pipeline: Recognition extracts
multimodal features through adaptive fidelity control, Memory aggregates events to compress 200+ dialogue turns into
50-80 semantic
units, and Attribution performs cross-modal alignment and progressive reasoning to identify emotional causal chains. The
framework
transforms multimodal dialogue sequences into structured representations processable by text-based LLMs while preserving
long-distance causal
dependencies.

![structure.png](resources/structure.png)

# Result

Impact of MCF on LLMs Performance Across Different Evaluation Metrics. The scores are reported in percentage (%). ↓and +
indicate performance decrease and increase compared to GPT-o1(text-only), respectively. Bold values represent the best
performance in each
section. Causemotion uses only the text modal. LLM refers to using only the LLM itself.

![bench.png](resources/bench.png)

## Quick Start

1. Make sure the following models are installed:

+ google-bert/bert-base-uncased
+ google/siglip-base-patch16-224
+ openai/whisper-large-v3
+ StarJiaxing/R1-Omni-0.5B
+ Qwen/Qwen2-Audio-7B-Instruct

Change the corresponding model paths in R1-Omni-0.5B/config.json (lines 23 and 31):

```
"mm_audio_tower": "/path/to/local/models/whisper-large-v3",
"mm_vision_tower": "/path/to/local/models/siglip-base-patch16-224"
```

1. install the required packages:

```shell
pip install -r requirements.txt
```

1. In the `main` folder, run

```shell
git clone https://github.com/HumanMLLM/HumanOmni
git clone https://github.com/StarsfieldAI/R1-V
```

and keep only the `humanomni` folder. this folder will looks like

```
.
├── R1-V
├── audio.py
├── audio_convert.py
├── combined.py
├── get_emo_score.py
├── get_emo_sw.py
├── humanomni # this is humanomni folder
├── utils.py
└── video.py
```

Then in `vhumanomni_arch.py`, replace the actual path to BERT:

```python
bert_model = "/gpfs/work/aac/yulongli19/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased"  # change to your BERT model path
```

1. Now to run:

+ We provide `convert_text` for text transcription:

    ```
    cd convert_text
    python main.py --input {chat.mp4} --api-key --parallel 8 --api-mode high
    ```

+ using with `audio_convert.py` for audio extraction:
  
  ```
    cd main
    python audio_convert.py --input_dir {mp4} --output_dir {mp3}
    ```
  
+ Once all three files (mp3, mp4, txt) are ready, first perform audio feature extraction:

    ```
  python audio.py --audio_path --model_path --output_path
    ```
  
+ Then extract video features:
  
  ```shell
    ce convert_text
  python video.py --root_dir {data} --output_dir --modal {audio or video_audio}
    ```
  
+ Concatenate the extracted content:
  
    ```shell
  python combined.py --audio_dir {audio} --input_dir {video_info} --output_dir {output}
  ```

1. After all feature extraction is done, generate the causal chains:

```shell
python get_emo_sw.py --input_dir --other_text {combined_text} --output_dir --config_path --llm_model --batch --window_sizes --step_sizes
```

1. To get the benchmark, run as:

```shell
python get_emo_score.py --gt_dir --input_dir --output_dir --batch --event_threshold
```
