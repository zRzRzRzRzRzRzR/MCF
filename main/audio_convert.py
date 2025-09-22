import argparse
import os

from moviepy import VideoFileClip


def convert_mp4_to_mp3(mp4_file, output_mp3=None):
    if not os.path.isfile(mp4_file):
        print(f"Error: File '{mp4_file}' does not exist.")
        return

    if not mp4_file.lower().endswith(".mp4"):
        print(f"Error: '{mp4_file}' is not an MP4 file.")
        return

    if output_mp3 is None:
        base_name = os.path.splitext(mp4_file)[0]
        output_mp3 = base_name + ".mp3"
    else:
        if not output_mp3.lower().endswith(".mp3"):
            output_mp3 += ".mp3"

    try:
        video = VideoFileClip(mp4_file)
        audio = video.audio
        audio.write_audiofile(output_mp3)
        audio.close()
        video.close()
    except Exception as e:
        print(f"Conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP4 video file to MP3 audio file"
    )
    parser.add_argument("input", help="Input MP4 file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output MP3 file path (optional, defaults to same directory with .mp3 extension)",
    )

    args = parser.parse_args()

    convert_mp4_to_mp3(args.input, args.output)


if __name__ == "__main__":
    main()
