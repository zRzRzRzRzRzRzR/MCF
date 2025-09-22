import argparse
import json
import os


def merge_json_files(
    audio_analysis_path: str, emotion_timeline_path: str, output_path: str
) -> None:
    with open(audio_analysis_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)
    with open(emotion_timeline_path, "r", encoding="utf-8") as f:
        emotion_data = json.load(f)
    merged_data = {**audio_data, **emotion_data}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully merged file: {output_path}")


def process_batch(audio_dir: str, emotion_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith("_audio.json")]

    if not audio_files:
        print(f"No audio analysis files found in {audio_dir}")
        return

    for audio_file in audio_files:
        base_name = audio_file.replace("_audio.json", "")
        audio_path = os.path.join(audio_dir, audio_file)
        emotion_path = os.path.join(emotion_dir, f"{base_name}_emotion.json")
        output_path = os.path.join(output_dir, f"{base_name}_merged.json")
        if not os.path.exists(emotion_path):
            continue
        merge_json_files(audio_path, emotion_path, output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine audio analysis and emotion timeline results"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio analysis files (format: *_audio.json)",
    )
    parser.add_argument(
        "--emotion_dir",
        type=str,
        required=True,
        help="Directory containing emotion timeline files (format: *_emotion.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged files (format: *_merged.json)",
    )

    args = parser.parse_args()
    process_batch(args.audio_dir, args.emotion_dir, args.output_dir)


if __name__ == "__main__":
    main()
