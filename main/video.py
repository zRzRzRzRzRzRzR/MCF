import os
import argparse
import re
import json
from humanomni import model_init, mm_infer
from main.humanomni.utils import disable_torch_init
from transformers import BertTokenizer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def extract_speaker_data(file_path):
    speaker_data = []
    current_speaker = None
    current_timestamp = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            speaker_match = re.match(r'发言人\s*(\d+)\s*(\d{2}:\d{2})', line)
            if speaker_match:
                if current_speaker is not None:
                    speaker_data.append({
                        'speaker': current_speaker,
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text)
                    })

                current_speaker = speaker_match.group(1)
                current_timestamp = speaker_match.group(2)
                current_text = []
            elif line and not line.startswith('[') and not line.startswith('http'):
                current_text.append(line)

        if current_speaker is not None:
            speaker_data.append({
                'speaker': current_speaker,
                'timestamp': current_timestamp,
                'text': ' '.join(current_text)
            })
    
    return speaker_data

def format_prompt(speaker_data):
    prompt_parts = []
    for entry in speaker_data:
        prompt_parts.append(
            f"At {entry['timestamp']}, Speaker {entry['speaker']} said: {entry['text']}"
        )
    
    base_instruct = "please analysis each speakers emotion,and records. Output the thinkong process in  and final emotion in <answer> </answer> tags."
    return "Here is a conversation transcript with timestamps and speakers:\n" + "\n".join(prompt_parts) + "\n\n" + base_instruct

def process_folder(folder_path, output_dir, model, processor, tokenizer, bert_tokenizer, modal="video_audio"):
    folder_name = os.path.basename(folder_path)
    print(f"Processing folder: {folder_name}")

    video_path = os.path.join(folder_path, f"{folder_name}.mp4")
    transcript_file = os.path.join(folder_path, f"{folder_name}.txt")

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
        return
    if not os.path.exists(transcript_file):
        print(f"Warning: Transcript file not found at {transcript_file}")
        return
    

    speaker_data = extract_speaker_data(transcript_file)
    instruct = format_prompt(speaker_data)
    video_tensor = processor['video'](video_path)
    
    if modal == 'video_audio' or modal == 'audio':
        audio = processor['audio'](video_path)[0]
    else:
        audio = None

    output = mm_infer(
        video_tensor, 
        instruct, 
        model=model, 
        tokenizer=tokenizer, 
        modal=modal, 
        question=instruct, 
        bert_tokeni=bert_tokenizer, 
        do_sample=False, 
        audio=audio
    )

    output_file = os.path.join(output_dir, f"{folder_name}_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch process chat folders for emotion analysis")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing chat-<number> folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON files")
    parser.add_argument("--modal", type=str, default="video_audio", choices=["video_audio", "audio", "video"], 
                        help="Modal type for processing")
    parser.add_argument("--bert_model", type=str, default="video_audio", choices=["video_audio", "audio", "video"],
                        help="Modal type for processing")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    disable_torch_init()
    model_path = ""
    model, processor, tokenizer = model_init(model_path)
    
    chat_folders = []
    for item in os.listdir(args.root_dir):
        item_path = os.path.join(args.root_dir, item)
        if os.path.isdir(item_path) and re.match(r'chat-\d+', item):
            chat_folders.append(item_path)
    
    if not chat_folders:
        print("No chat-<number> folders found in the specified directory")
        return
    
    print(f"Found {len(chat_folders)} chat folders to process")
    
    # 处理每个文件夹
    for folder_path in chat_folders:
        try:
            process_folder(
                folder_path=folder_path,
                output_dir=args.output_dir,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                bert_tokenizer=bert_tokenizer,
                modal=args.modal
            )
        except Exception as e:
            print(f"Error processing folder {folder_path}: {str(e)}")
            continue
    
    print(f"\nBatch processing completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
