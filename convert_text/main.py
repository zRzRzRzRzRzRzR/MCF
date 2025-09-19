import argparse
import os
import sys
import glob
import time
from error_detector import ErrorDetector
from config import Config

def find_transcript_files(input_path: str, recursive: bool = False) -> list:
    files = []
    
    if os.path.isfile(input_path):
        files.append(input_path)
    elif os.path.isdir(input_path):
        pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(input_path, pattern)
        files.extend(glob.glob(search_path, recursive=recursive))
    else:
        files.extend(glob.glob(input_path))

    valid_files = []
    for file in files:
        if (file.endswith('.txt') and
            not file.endswith('_corrected.txt') and
            not file.endswith('_report.txt') and
            os.path.isfile(file)):
            valid_files.append(file)

    return valid_files

def configure_high_api_usage(detector: ErrorDetector, mode: str = 'high'):
    if not hasattr(detector, 'glm_client'):
        print("⚠️ No GLM client on detector; skipping high-API configuration")
        return

    api_configs = {
        'maximum': {
            'api_usage_ratio': 0.95,
            'max_tokens': 500,
            'use_detailed_prompts': True,
            'min_text_length': 2,
            'description': 'Maximum API usage mode — expected 30K+ tokens'
        },
        'high': {
            'api_usage_ratio': 0.80,
            'max_tokens': 400,
            'use_detailed_prompts': True,
            'min_text_length': 5,
            'description': 'High API usage mode — expected 20K+ tokens'
        },
        'medium': {
            'api_usage_ratio': 0.50,
            'max_tokens': 300,
            'use_detailed_prompts': False,
            'min_text_length': 8,
            'description': 'Medium API usage mode — expected 10K+ tokens'
        }
    }

    config = api_configs.get(mode, api_configs['high'])

    detector.glm_client.high_api_mode = True
    detector.glm_client.api_usage_ratio = config['api_usage_ratio']
    detector.glm_client.max_tokens = config['max_tokens']
    detector.glm_client.use_detailed_prompts = config['use_detailed_prompts']
    detector.glm_client.min_text_length = config['min_text_length']

    print(f"✅ Enabled {mode.upper()} mode: {config['description']}")
    print(f"   API usage ratio: {config['api_usage_ratio']*100:.0f}%")
    print(f"   Max tokens per call: {config['max_tokens']}")
    print(f"   Detailed prompts: {config['use_detailed_prompts']}")

def process_single_file(detector: ErrorDetector, file_path: str, args) -> dict:
    try:
        print(f"\n📄 Processing file: {file_path}")
        start_time = time.time()

        if args.only_correct:
            corrected_path = detector.detect_and_correct_file_only_correct(file_path)
            report_path = None
        else:
            report_path, corrected_path = detector.detect_and_correct_file(file_path)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            'file': file_path,
            'status': 'success',
            'report_path': report_path,
            'corrected_path': corrected_path,
            'processing_time': processing_time,
            'error': None
        }

    except Exception as e:
        return {
            'file': file_path,
            'status': 'error',
            'report_path': None,
            'corrected_path': None,
            'processing_time': 0,
            'error': str(e)
        }

def generate_batch_summary(results: list, output_dir: str) -> str:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")

    total_files = len(results)
    successful_files = len([r for r in results if r['status'] == 'success'])
    failed_files = total_files - successful_files
    total_time = sum(r['processing_time'] for r in results)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("批量处理总结报告 (修复优化版)\n")
        f.write("=" * 70 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"成功处理: {successful_files}\n")
        f.write(f"处理失败: {failed_files}\n")
        f.write(f"成功率: {successful_files/total_files*100:.1f}%\n")
        f.write(f"总耗时: {total_time:.1f}秒\n")
        f.write(f"平均耗时: {total_time/total_files:.1f}秒/文件\n\n")

        f.write("=" * 70 + "\n")
        f.write("详细处理结果\n")
        f.write("=" * 70 + "\n\n")

        if successful_files > 0:
            f.write("✅ 成功处理的文件:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                if result['status'] == 'success':
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"耗时: {result['processing_time']:.1f}秒\n")
                    if result['report_path']:
                        f.write(f"报告: {result['report_path']}\n")
                    f.write(f"修正: {result['corrected_path']}\n\n")

        if failed_files > 0:
            f.write("\n❌ 处理失败的文件:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                if result['status'] == 'error':
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"错误: {result['error']}\n\n")

    return summary_path

def main():
    parser = argparse.ArgumentParser(description='Speech transcription error detection and correction tool - supports batch processing')
    parser.add_argument('input', help='Input file/directory path or wildcard pattern (e.g. "*.txt" or "transcripts/")')
    parser.add_argument('--api-key', help='GLM API key (optional, environment variable has priority)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively process files in subdirectories')
    parser.add_argument('--correct', action='store_true', help='Also generate corrected version')
    parser.add_argument('--only-correct', action='store_true', help='Generate corrected version only, no detection report')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection')
    parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel threads (default: serial processing)')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue processing other files when an error occurs')
    parser.add_argument('--dry-run', action='store_true', help='Dry-run: only list files to be processed, no actual processing')

    parser.add_argument('--api-mode', choices=['low', 'medium', 'high', 'maximum'],
                       default='high', help='API usage mode (default: high)')

    args = parser.parse_args()

    try:
        api_key = args.api_key or Config.GLM_API_KEY
        detector = ErrorDetector(api_key)

        print(f"\n🚀 Configuring API usage mode...")
        configure_high_api_usage(detector, args.api_mode)

        if args.test_connection:
            print("\n🔍 Testing API connection...")
            if detector.glm_client.test_connection():
                print("✅ API connection OK")
            else:
                print("❌ API connection failed; please check your key and network")
                sys.exit(1)
            return

        print(f"\n🔍 Searching transcript files: {args.input}")
        files = find_transcript_files(args.input, args.recursive)

        if not files:
            print(f"❌ No matching transcript files found: {args.input}")
            sys.exit(1)

        print(f"📋 Found {len(files)} files to process")

        for i, file in enumerate(files, 1):
            print(f"  {i}. {file}")

        if args.dry_run:
            print("\n🔍 Dry-run: the above files would be processed")
            print("Using --dry-run only previews; no files will be processed")
            return

        if len(files) > 1:
            print(f"\n📊 Batch run summary:")
            print(f"   Files: {len(files)}")
            print(f"   Mode: {'corrections only' if args.only_correct else 'report + corrections'}")
            print(f"   API mode: {args.api_mode.upper()}")
            print(f"   API key: {api_key[:8]}...{api_key[-4:]}")

            confirm = input("\nStart batch processing? (y/N): ").lower().strip()
            if confirm not in ['y', 'yes']:
                print("❌ User cancelled")
                return

        if len(files) > 1:
            print(f"\n🚀 Starting batch processing of {len(files)} files...")
        else:
            print(f"\n🚀 Starting processing...")
        print("=" * 60)

        results = []

        if args.parallel and args.parallel > 1:
            print(f"📄 Using {args.parallel} threads for parallel processing...")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_file = {
                    executor.submit(process_single_file, detector, file, args): file
                    for file in files
                }

                for i, future in enumerate(as_completed(future_to_file), 1):
                    file = future_to_file[future]
                    print(f"📊 Progress: {i}/{len(files)} done")

                    try:
                        result = future.result()
                        results.append(result)

                        if result['status'] == 'success':
                            print(f"✅ {result['file']} - succeeded")
                        else:
                            print(f"❌ {result['file']} - failed: {result['error']}")
                            if not args.continue_on_error:
                                print("⚠️ Aborting (use --continue-on-error to continue with other files)")
                                break

                    except Exception as e:
                        print(f"❌ {file} - exception: {str(e)}")
                        results.append({
                            'file': file, 'status': 'error', 'error': str(e),
                            'report_path': None, 'corrected_path': None, 'processing_time': 0
                        })
        else:
            for i, file in enumerate(files, 1):
                if len(files) > 1:
                    print(f"\n📊 Progress: {i}/{len(files)}")

                result = process_single_file(detector, file, args)
                results.append(result)

                if result['status'] == 'success':
                    print(f"✅ Done ({result['processing_time']:.1f}s)")
                    if result['report_path']:
                        print(f"📊 Report: {result['report_path']}")
                    print(f"📝 Corrected: {result['corrected_path']}")
                else:
                    print(f"❌ Failed: {result['error']}")
                    if not args.continue_on_error:
                        print("⚠️ Aborting (use --continue-on-error to continue with other files)")
                        break

        if len(files) > 1:
            summary_path = generate_batch_summary(results, Config.OUTPUT_DIR)
            print(f"\n📈 Batch summary: {summary_path}")

        successful = len([r for r in results if r['status'] == 'success'])
        failed = len(results) - successful
        total_time = sum(r['processing_time'] for r in results)

        print("\n" + "=" * 60)
        if len(files) > 1:
            print("🎉 Batch processing complete!")
        else:
            print("🎉 Processing complete!")
        print("=" * 60)
        print(f"📊 Stats:")
        print(f"   Total files: {len(results)}")
        print(f"   Succeeded: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {successful/len(results)*100:.1f}%")
        print(f"   Total time: {total_time:.1f}s")
        if successful > 0:
            print(f"   Avg time: {total_time/successful:.1f}s/file")

        if successful > 0:
            print(f"\n📁 Output files saved to: {Config.OUTPUT_DIR}")

        if failed > 0:
            sys.exit(1)

    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()