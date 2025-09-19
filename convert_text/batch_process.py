import os
import sys
import glob

def show_menu():
    """显示操作菜单"""
    print("\n" + "="*60)
    print("🚀 Batch Correction Tool for Transcription Text (Large-batch Optimized)")
    print("="*60)
    print()
    print("Select an action:")
    print()
    print("1. 📁 Process all txt files in the current directory (recommended batch mode)")
    print("2. 📂 Process a specified directory (including subdirectories)")
    print("3. 🔍 Use a wildcard pattern (e.g., *.txt)")
    print("4. 📄 Process a single file")
    print("5. 🔧 Custom advanced options")
    print("6. 🧪 Test API connection")
    print("7. 📊 Estimate processing cost (optimized)")
    print("0. Exit")
    print()

def get_processing_mode():
    """获取处理模式选择"""
    print("Processing mode:")
    print("1. 📊 Full mode (recommended) — generate detailed report and corrected file")
    print("2. ⚡ Fast mode — generate corrected file only (faster)")

    while True:
        mode = input("Choose mode (1/2, Enter for Full): ").strip()
        if not mode:
            mode = "1"

        if mode in ["1", "2"]:
            return mode
        else:
            print("❌ Please enter 1 or 2")

def estimate_processing_cost(files):
    """
    预估处理成本和时间 - 基于大批次优化
    """
    if not files:
        return

    print(f"\n📊 Estimated processing cost (large-batch optimized)")
    print("-" * 50)

    total_files = len(files)

    total_size = 0
    estimated_segments = 0
    sample_files = files[:min(10, len(files))]

    for file in sample_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                file_size = len(content)
                total_size += file_size

                lines = content.strip().split('\n')
                segments = 0
                for line in lines:
                    line = line.strip()
                    if (line and
                        not line.startswith('=') and
                        not line.startswith('-') and
                        not line.startswith('chat-') and
                        len(line) > 10):
                        if ('发言人' in line and ':' in line) or '[' in line or len(line) > 20:
                            segments += 1

                estimated_segments += segments
        except:
            continue

    if len(sample_files) > 0:
        avg_segments = estimated_segments / len(sample_files)
        total_estimated_segments = int(avg_segments * total_files)
    else:
        total_estimated_segments = 0

    if total_estimated_segments == 0:
        print("❌ Unable to estimate segments; please check file format")
        return

    batch_size = 30

    quick_fixes_estimated = int(total_estimated_segments * 0.08)
    pre_filter_estimated = int(total_estimated_segments * 0.50)
    api_segments = total_estimated_segments - quick_fixes_estimated - pre_filter_estimated

    api_batches = max(1, (api_segments + batch_size - 1) // batch_size)

    base_prompt_tokens = 180
    tokens_per_segment = 25
    response_tokens_per_segment = 30

    total_input_tokens = 0
    total_output_tokens = 0

    for batch_idx in range(api_batches):
        segments_in_batch = min(batch_size, api_segments - batch_idx * batch_size)

        batch_input_tokens = base_prompt_tokens + (segments_in_batch * tokens_per_segment)
        batch_output_tokens = segments_in_batch * response_tokens_per_segment

        total_input_tokens += batch_input_tokens
        total_output_tokens += batch_output_tokens

    total_tokens_estimated = total_input_tokens + total_output_tokens

    api_batch_time = api_batches * 3.0
    quick_fix_time = quick_fixes_estimated * 0.005
    pre_filter_time = pre_filter_estimated * 0.001
    total_time = api_batch_time + quick_fix_time + pre_filter_time

    print(f"Total files: {total_files}")
    print(f"Estimated segments: {total_estimated_segments:,}")
    print()
    print("Large-batch processing distribution:")
    print(f"  API batches: {api_batches} ({batch_size} segments per batch)")
    print(f"  API-processed segments: {api_segments:,} ({api_segments/total_estimated_segments*100:.1f}%)")
    print(f"  Quick fixes: {quick_fixes_estimated:,} ({quick_fixes_estimated/total_estimated_segments*100:.1f}%)")
    print(f"  Pre-filter skipped: {pre_filter_estimated:,} ({pre_filter_estimated/total_estimated_segments*100:.1f}%)")
    print()
    print(f"Token usage analysis:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_tokens_estimated:,}")
    print(f"Estimated time: {total_time/60:.1f} minutes")

    input_cost_per_1k = 0.0005
    output_cost_per_1k = 0.002
    estimated_cost = (total_input_tokens / 1000) * input_cost_per_1k + (total_output_tokens / 1000) * output_cost_per_1k
    print(f"Estimated cost: ¥{estimated_cost:.3f}")

    original_api_calls = int(total_estimated_segments * 0.42)
    original_tokens = original_api_calls * 1000
    original_cost = (original_tokens / 1000) * 0.002

    print()
    print("🔥 Large-batch optimization comparison:")
    print(f"  Original API calls: {original_api_calls} times")
    print(f"  Optimized API calls: {api_batches} times")
    print(f"  Calls reduced: {(1 - api_batches/max(original_api_calls, 1))*100:.1f}%")
    print(f"  Tokens reduced: {(1 - total_tokens_estimated/max(original_tokens, 1))*100:.1f}%")
    print(f"  Cost reduced: {(1 - estimated_cost/max(original_cost, 0.001))*100:.1f}%")
    print(f"  Time reduced: ~90-95%")

    if total_files > 20:
        print(f"\n💡 Tips:")
        print(f"  • Large-batch processing is highly optimized; you can run directly")
        print(f"  • Use Full mode to get a detailed report")
        print(f"  • Optionally process a small batch first to validate results")

    if total_estimated_segments > 1000:
        print(f"  • Many segments detected, but large-batch mode greatly optimizes time and cost")
        print(f"  • Estimated time: {total_time/60:.1f} minutes, cost: ¥{estimated_cost:.3f}")

def get_file_count(pattern, recursive=False):
    if os.path.isfile(pattern):
        return 1
    elif os.path.isdir(pattern):
        search_pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(pattern, search_pattern)
        files = glob.glob(search_path, recursive=recursive)
        files = [f for f in files if not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
        return len(files)
    else:
        files = glob.glob(pattern)
        files = [f for f in files if f.endswith('.txt') and not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
        return len(files)

def get_files_list(pattern, recursive=False):
    files = []
    if os.path.isfile(pattern):
        files.append(pattern)
    elif os.path.isdir(pattern):
        search_pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(pattern, search_pattern)
        files.extend(glob.glob(search_path, recursive=recursive))
    else:
        files.extend(glob.glob(pattern))

    valid_files = [f for f in files if f.endswith('.txt') and not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
    return valid_files

def run_processing(command):
    print(f"\n🚀 Running command: {command}")
    print("-" * 60)

    confirm = input("Start processing? (y/N): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return

    exit_code = os.system(command)

    if exit_code == 0:
        print("\n🎉 Processing complete!")
    else:
        print(f"\n❌ Processing failed, exit code: {exit_code}")

def main():
    while True:
        show_menu()

        try:
            choice = input("Select an option (0–7): ").strip()

            if choice == '0':
                print("👋 Goodbye!")
                break

            elif choice == '1':
                file_count = get_file_count(".")
                if file_count == 0:
                    print("❌ No txt files found in current directory")
                    continue

                print(f"📊 Found {file_count} files")

                mode = get_processing_mode()

                print("Large-batch processing is highly optimized; no extra parallel settings needed")

                if mode == '2':
                    command = f"python main.py . --only-correct --continue-on-error"
                else:
                    command = f"python main.py . --correct --continue-on-error"

                run_processing(command)

            elif choice == '2':
                directory = input("Enter directory path: ").strip()
                if not directory:
                    print("❌ Please enter a valid directory path")
                    continue

                if not os.path.isdir(directory):
                    print("❌ Directory not found")
                    continue

                file_count = get_file_count(directory, recursive=True)
                if file_count == 0:
                    print("❌ No txt files found in the specified directory")
                    continue

                print(f"📊 Found {file_count} files (including subdirectories)")

                mode = get_processing_mode()

                if mode == '2':
                    command = f'python main.py "{directory}" --only-correct --recursive --continue-on-error'
                else:
                    command = f'python main.py "{directory}" --correct --recursive --continue-on-error'

                run_processing(command)

            elif choice == '3':
                pattern = input("Enter a wildcard pattern (e.g., *.txt or chat-*.txt): ").strip()
                if not pattern:
                    print("❌ Please enter a valid wildcard pattern")
                    continue

                file_count = get_file_count(pattern)
                if file_count == 0:
                    print("❌ No matching files found")
                    continue

                print(f"📊 Found {file_count} matching files")

                preview = input("Preview matching files? (y/N): ").lower().strip()
                if preview in ['y', 'yes']:
                    os.system(f'python main.py "{pattern}" --dry-run')

                mode = get_processing_mode()

                if mode == '2':
                    command = f'python main.py "{pattern}" --only-correct --continue-on-error'
                else:
                    command = f'python main.py "{pattern}" --correct --continue-on-error'

                run_processing(command)

            elif choice == '4':
                file_path = input("Enter file path: ").strip()
                if not file_path:
                    print("❌ Please enter a valid file path")
                    continue

                if not os.path.isfile(file_path):
                    print("❌ File not found")
                    continue

                mode = get_processing_mode()

                if mode == '2':
                    command = f'python main.py "{file_path}" --only-correct'
                else:
                    command = f'python main.py "{file_path}" --correct'

                run_processing(command)

            elif choice == '5':
                print("\n🔧 Custom advanced options")
                print("Recommended large-batch command templates:")
                print("python main.py \"*.txt\" --correct --continue-on-error  # Full mode")
                print("python main.py \"*.txt\" --only-correct --continue-on-error  # Fast mode")
                print("python main.py transcripts/ --recursive --correct  # Directory full mode")
                print()

                custom_command = input("Enter command: ").strip()
                if not custom_command:
                    print("❌ Please enter a valid command")
                    continue

                if not custom_command.startswith('python main.py'):
                    print("❌ Command must start with 'python main.py'")
                    continue

                run_processing(custom_command)

            elif choice == '6':
                print("\n🔍 Testing API connection...")
                test_file = "test.txt"
                if not os.path.exists(test_file):
                    with open("temp_test.txt", "w", encoding="utf-8") as f:
                        f.write("发言人1 00:01\n这是测试内容，我觉的应该没有错误。")
                    test_file = "temp_test.txt"

                os.system(f'python main.py "{test_file}" --test-connection')

                if test_file == "temp_test.txt":
                    try:
                        os.remove("temp_test.txt")
                    except:
                        pass

            elif choice == '7':
                pattern = input("Enter a file pattern or directory to analyze (e.g., *.txt or a folder path): ").strip()
                if not pattern:
                    pattern = "."

                files = get_files_list(pattern, recursive=True)
                if files:
                    estimate_processing_cost(files)
                else:
                    print("❌ No matching files found")

            else:
                print("❌ Invalid choice; please enter a number between 0 and 7")

        except KeyboardInterrupt:
            print("\n⚠️  User interrupted")
            break
        except Exception as e:
            print(f"❌ Operation failed: {e}")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found")
        print("Please run this script in a directory that contains main.py")
        sys.exit(1)
    
    main()