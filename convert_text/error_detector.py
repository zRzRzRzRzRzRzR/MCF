import os
import time
from datetime import datetime
from typing import Dict, List

from config import Config
from glm_client import GLMClient
from text_processor import TextProcessor


class ErrorDetector:
    def __init__(self, api_key: str = None):
        self.glm_client = GLMClient(api_key)
        self.text_processor = TextProcessor()
        self.results = []

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

    def detect_and_correct_file(self, input_file: str) -> tuple:
        print(f"Starting to process file: {input_file}")

        segments = self.text_processor.parse_transcription_file(input_file)
        print(f"Parsed {len(segments)} segments")

        valid_segments = [seg for seg in segments if seg.get("text", "").strip()]
        print(f"Valid segments: {len(valid_segments)}")

        print("Starting error detection and auto-correction...")
        print("Using batch mode to greatly reduce API calls and token usage...")

        start_time = time.time()
        results = self.glm_client.batch_detect_and_correct_segments(valid_segments)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Batch processing completed in {processing_time:.1f}s")

        report_path = self._generate_correction_report(results, input_file)

        corrected_path = self._generate_corrected_file(results, input_file)

        self._print_correction_summary(results)

        print("Processing complete!")
        print(f"📊 Report: {report_path}")
        print(f"📝 Corrected file: {corrected_path}")

        return report_path, corrected_path

    def _generate_correction_report(self, results: List[Dict], input_file: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(input_file))[0]
        report_path = os.path.join(
            Config.OUTPUT_DIR, f"{filename}_correction_report_{timestamp}.txt"
        )

        total_segments = len(results)
        corrected_segments = sum(1 for r in results if r.get("has_errors", False))
        api_errors = sum(1 for r in results if "error" in r)

        batch_api_count = len([r for r in results if r.get("method") == "batch_api"])
        quick_fix_count = len([r for r in results if r.get("method") == "quick_fix"])
        pre_filter_count = len([r for r in results if r.get("method") == "pre_filter"])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(
                "Automatic Correction Report for Speech Transcription (Batch-Optimized)\n"
            )
            f.write("=" * 70 + "\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total segments: {total_segments}\n")
            f.write(f"Corrected segments: {corrected_segments}\n")
            f.write(
                f"Correction rate: {corrected_segments / total_segments * 100:.2f}%\n"
            )
            if api_errors > 0:
                f.write(f"API errors: {api_errors}\n")

            f.write("\nProcessing method distribution:\n")
            f.write(
                f"  Batch API processing: {batch_api_count} ({batch_api_count / total_segments * 100:.1f}%)\n"
            )
            f.write(
                f"  Quick fix: {quick_fix_count} ({quick_fix_count / total_segments * 100:.1f}%)\n"
            )
            f.write(
                f"  Pre-filter skipped: {pre_filter_count} ({pre_filter_count / total_segments * 100:.1f}%)\n"
            )

            f.write("\n" + "=" * 70 + "\n")
            f.write("Detailed corrections\n")
            f.write("=" * 70 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"[Segment {i}]\n")
                f.write(f"Time: {result.get('timestamp', 'Unknown')}\n")
                f.write(f"Speaker: {result.get('speaker', 'Unknown')}\n")
                f.write(f"Method: {result.get('method', 'Unknown')}\n")

                if "error" in result:
                    f.write(f"❌ API call error: {result['error']}\n")
                    f.write(f"Original: {result.get('text', '')}\n")

                elif result.get("has_errors", False):
                    f.write("🔧 Corrected\n")
                    f.write(
                        f"Original: {result.get('original_text', result.get('text', ''))}\n"
                    )
                    f.write(f"Fix: {result.get('corrected_text', '')}\n")
                    f.write(f"Confidence: {result.get('confidence', 0):.2f}\n")

                    errors = result.get("errors", [])
                    if errors:
                        f.write("Error details:\n")
                        for j, error in enumerate(errors, 1):
                            f.write(f"  {j}. {error.get('type', 'Unknown')}: ")
                            f.write(
                                f"'{error.get('original', '')}' → '{error.get('corrected', '')}'\n"
                            )
                            if error.get("reason"):
                                f.write(f"     Reason: {error.get('reason')}\n")

                else:
                    f.write("✅ No correction needed\n")
                    f.write(f"Text: {result.get('text', '')}\n")

                f.write("\n" + "-" * 50 + "\n\n")

        return report_path

    def _generate_corrected_file(self, results: List[Dict], input_file: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(input_file))[0]
        corrected_path = os.path.join(
            Config.OUTPUT_DIR, f"{filename}_corrected_{timestamp}.txt"
        )

        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(f"{filename} - Auto-corrected Version (Batch-Optimized)\n\n")
            f.write(f"Corrected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original file: {input_file}\n")
            f.write("=" * 50 + "\n\n")

            for result in results:
                speaker = result.get("speaker", "Unknown")
                timestamp = result.get("timestamp", "Unknown")

                if "error" in result:
                    display_text = result.get("text", "[Processing error]")
                    f.write(f"{speaker} {timestamp}\n")
                    f.write(f"❌ {display_text}\n\n")

                elif result.get("has_errors", False) and result.get("corrected_text"):
                    corrected_text = result.get(
                        "corrected_text", result.get("text", "")
                    )
                    if "\n" in corrected_text:
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{corrected_text}\n\n")
                    else:
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{corrected_text}\n\n")

                else:
                    original_text = result.get("text", "")
                    if original_text.strip():
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{original_text}\n\n")

        return corrected_path

    def _print_correction_summary(self, results: List[Dict]):
        total = len(results)
        corrected = sum(1 for r in results if r.get("has_errors", False))
        errors = sum(1 for r in results if "error" in r)
        unchanged = total - corrected - errors

        batch_api_count = len([r for r in results if r.get("method") == "batch_api"])
        quick_fix_count = len([r for r in results if r.get("method") == "quick_fix"])
        pre_filter_count = len([r for r in results if r.get("method") == "pre_filter"])

        print("\n" + "=" * 50)
        print("📊 Correction Summary")
        print("=" * 50)
        print(f"Total segments: {total}")
        print(f"Corrected: {corrected} ({corrected / total * 100:.1f}%)")
        print(f"Unchanged: {unchanged} ({unchanged / total * 100:.1f}%)")
        if errors > 0:
            print(f"Failed: {errors} ({errors / total * 100:.1f}%)")

        print("\nBatch processing optimization:")
        print(f"Batch API: {batch_api_count} ({batch_api_count / total * 100:.1f}%)")
        print(f"Quick fix: {quick_fix_count} ({quick_fix_count / total * 100:.1f}%)")
        print(
            f"Pre-filter skipped: {pre_filter_count} ({pre_filter_count / total * 100:.1f}%)"
        )
        print("=" * 50)

    def detect_and_correct_file_only_correct(self, input_file: str) -> str:
        print(f"Starting to process file: {input_file}")

        segments = self.text_processor.parse_transcription_file(input_file)
        print(f"Parsed {len(segments)} segments")

        valid_segments = [seg for seg in segments if seg.get("text", "").strip()]
        print(f"Valid segments: {len(valid_segments)}")

        print(
            "Starting error detection and auto-correction (only generate corrected file)..."
        )

        start_time = time.time()
        results = self.glm_client.batch_detect_and_correct_segments(valid_segments)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Batch processing completed in {processing_time:.1f}s")

        corrected_path = self._generate_corrected_file(results, input_file)

        self._print_correction_summary(results)

        print("Processing complete!")
        print(f"📝 Corrected file: {corrected_path}")

        return corrected_path
