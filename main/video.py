import os
import argparse
import re
import json
import torch
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def extract_speaker_timestamps(file_path):
    """Extract the speaker and timestamp information from the txt file"""
    
    pattern = re.compile(r'发言人\s+\d+\s+\d{2}:\d{2}')
    
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            matches = pattern.findall(line)
            results.extend(matches)
    
    return results

def build_instruct(speaker_timestamps):
    """Model construction instructions"""
    # Combine all speakers and timestamps into instructions
    instruct = "\n".join(speaker_timestamps)
    # Add requirements for the model output format
    instruct += "\n\nPlease analyze the emotion for each speaker at each timestamp and output in the following format:\n"
    instruct += "[speaker timestamp emotion]"
    return instruct

def process_video(video_path, instruct, model, processor, tokenizer, bert_tokenizer, modal="video_audio"):
    """Process videos and perform model inference"""
    # Process video input
    video_tensor = processor['video'](video_path)
    
    # Decide whether to process the audio based on the modal type
    if modal == 'video_audio' or modal == 'audio':
        audio = processor['audio'](video_path)[0]
    else:
        audio = None

    # Carrying out reasoning
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
    
    return output

def process_folder(folder_path, output_dir, model, processor, tokenizer, bert_tokenizer, modal="video_audio"):
    """Handle a single folder"""
    folder_name = os.path.basename(folder_path)
    
    video_path = os.path.join(folder_path, f"{folder_name}.mp4")
    transcript_file = os.path.join(folder_path, f"{folder_name}.txt")
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
        return
    if not os.path.exists(transcript_file):
        print(f"Warning: Transcript file not found at {transcript_file}")
        return
    
    speaker_timestamps = extract_speaker_timestamps(transcript_file)
    if not speaker_timestamps:
        print(f"Warning: No speaker/timestamp data found in {transcript_file}")
        return
    instruct = build_instruct(speaker_timestamps)
    # Process videos and perform reasoning
    try:
        output = process_video(
            video_path=video_path,
            instruct=instruct,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            bert_tokenizer=bert_tokenizer,
            modal=modal
        )
        
        # Save the original output
        raw_output_file = os.path.join(output_dir, f"{folder_name}_raw_output.txt")
        with open(raw_output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        # Output of the analytical model
        pattern = re.compile(r'\[([^\]]+)\s+(\d{2}:\d{2}:\d{2})\s+([^\]]+)\]')
        matches = pattern.findall(output)
        results = []
        
        for match in matches:
            speaker, timestamp, emotion = match
            results.append({
                'speaker': speaker,
                'timestamp': timestamp,
                'emotion': emotion.strip()
            })
        
        # Save the result after parsing
        output_file = os.path.join(output_dir, f"{folder_name}_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Processed {folder_name}: {len(results)} emotions detected")
        print(f"Raw output saved to {raw_output_file}")
        print(f"Structured results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")
        error_file = os.path.join(output_dir, f"{folder_name}_error.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(str(e))

def main():
    parser = argparse.ArgumentParser(description="Batch process chat folders for emotion analysis")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing chat-<number> folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--modal", type=str, default="video_audio", choices=["video_audio", "audio", "video"], 
                        help="Modal type for processing")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model name or path")
    parser.add_argument("--model_path", type=str, default="/root/.cache/modelscope/hub/models/iic/R1-Omni-0.5B",
                        help="Path to the model")
    
    args = parser.parse_args()

    
    # Create an output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    disable_torch_init()
    
    model, processor, tokenizer = model_init(args.model_path)
    
    chat_folders = []
    for item in os.listdir(args.root_dir):
        item_path = os.path.join(args.root_dir, item)
        if os.path.isdir(item_path) and re.match(r'chat-\d+', item):
            chat_folders.append(item_path)
    
    if not chat_folders:
        print("No chat-<number> folders found in the specified directory")
        return
    
    print(f"Found {len(chat_folders)} chat folders to process")
    
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
