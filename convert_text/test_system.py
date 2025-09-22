import os
import sys
import tempfile

from config import Config
from error_detector import ErrorDetector
from glm_client import GLMClient
from text_processor import TextProcessor


def test_config():
    print("🔧 Testing configuration load...")
    try:
        api_key = Config.GLM_API_KEY
        base_url = Config.GLM_BASE_URL
        print(
            f"✅ API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}"
        )
        print(f"✅ API base URL: {base_url}")
        return True
    except Exception as e:
        print(f"❌ Configuration load failed: {e}")
        return False


def test_glm_client():
    print("\n🤖 Testing GLM client (batch mode)...")
    try:
        client = GLMClient()

        test_texts = [
            "这是一个测试文本，没有错误。",
            "我觉的这个方案不错。",
            "在说一遍好吗？",
        ]

        print("Testing batch processing...")
        results = client.batch_detect_and_correct_texts(test_texts)

        if len(results) != len(test_texts):
            print(
                f"❌ Result count mismatch: expected {len(test_texts)}, got {len(results)}"
            )
            return False

        api_calls = 0
        quick_fixes = 0
        skipped = 0

        for i, result in enumerate(results):
            if "error" in result:
                print(f"❌ Text {i + 1} failed: {result['error']}")
                return False
            else:
                print(
                    f"✅ Text {i + 1}: {result.get('method', 'unknown')} - {result.get('has_errors', False)}"
                )

                method = result.get("method", "")
                if method in ["batch_api"]:
                    api_calls += 1
                elif method == "quick_fix":
                    quick_fixes += 1
                else:
                    skipped += 1

        print(
            f"Stats: API batch {api_calls}, quick fixes {quick_fixes}, skipped {skipped}"
        )
        print("✅ Batch processing works")
        return True

    except Exception as e:
        print(f"❌ GLM client test failed: {e}")
        return False


def test_text_processor():
    print("\n📝 Testing text processor...")
    try:
        processor = TextProcessor()

        test_content = """发言人1 04:49
这是第一段测试内容，包含一些可能的错误文字。

发言人2 05:13
我觉的这个方案不错，我们应该仔细考虑一下。

发言人1 05:30
好的，那我们在会议上讨论一下吧。
"""

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write(test_content)
            test_file = f.name

        segments = processor.parse_transcription_file(test_file)

        print(f"✅ Parsed {len(segments)} segments")
        for i, segment in enumerate(segments[:3]):
            print(
                f"   Segment {i + 1}: {segment['speaker']} ({segment['timestamp']}) - {segment['text'][:30]}..."
            )

        os.unlink(test_file)
        return True

    except Exception as e:
        print(f"❌ Text processor test failed: {e}")
        return False


def test_error_detector():
    print("\n🔍 Testing error detector (batch mode)...")
    try:
        detector = ErrorDetector()

        test_content = """发言人1 04:49
老师说我们应该好好学习，天天向上，但是我觉的有些困难。

发言人2 05:13
我同意你的看法，学习确实需要持之以恒的努力。

发言人3 05:30
在说一遍这个观点，我觉的我们需要在努力一点。
"""

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write(test_content)
            test_file = f.name

        print("   Running batch processing test...")
        report_path, corrected_path = detector.detect_and_correct_file(test_file)

        if os.path.exists(report_path) and os.path.exists(corrected_path):
            print("✅ Error detector working (batch mode)")
            print(f"   Report file: {os.path.basename(report_path)}")
            print(f"   Corrected file: {os.path.basename(corrected_path)}")

            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                if "批量API处理" in report_content or "快速修正" in report_content:
                    print("✅ Batch processing functioning correctly")
                else:
                    print("⚠️  Batch processing may not be functioning correctly")

            os.unlink(test_file)
            return True
        else:
            print("❌ Output files not generated")
            return False

    except Exception as e:
        print(f"❌ Error detector test failed: {e}")
        return False


def test_file_formats():
    print("\n📄 Testing support for different file formats...")
    processor = TextProcessor()

    formats = {
        "发言人+时间戳": """发言人1 04:49
这是测试内容

发言人2 05:13
这是另一段内容""",
        "时间戳+内容": """[00:04:49] 这是测试内容
[00:05:13] 这是另一段内容""",
        "时间戳+发言人+内容": """[00:04:49-00:05:13] 张三: 这是测试内容
[00:05:13-00:05:30] 李四: 这是另一段内容""",
    }

    all_passed = True
    for format_name, content in formats.items():
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt", encoding="utf-8"
            ) as f:
                f.write(content)
                test_file = f.name

            segments = processor.parse_transcription_file(test_file)

            if len(segments) > 0:
                print(f"✅ {format_name}: parsed {len(segments)} segments")
            else:
                print(f"❌ {format_name}: parse failed")
                all_passed = False

            os.unlink(test_file)

        except Exception as e:
            print(f"❌ {format_name}: test failed - {e}")
            all_passed = False

    return all_passed


def test_batch_optimization():
    print("\n🚀 Testing batch optimization...")
    try:
        client = GLMClient()

        test_texts = [
            f"这是第{i}段测试文本，我觉的应该没有错误。" for i in range(1, 11)
        ]

        print(f"   Testing batch processing of {len(test_texts)} text segments...")

        import time

        start_time = time.time()
        results = client.batch_detect_and_correct_texts(test_texts, batch_size=5)
        end_time = time.time()

        processing_time = end_time - start_time

        if len(results) == len(test_texts):
            methods = {}
            for result in results:
                method = result.get("method", "unknown")
                methods[method] = methods.get(method, 0) + 1

            print(f"✅ Batch processing succeeded")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Avg per segment: {processing_time / len(test_texts):.3f}s")
            print(f"   Methods distribution: {methods}")

            if "batch_api" in methods:
                print("✅ Batch API processing effective")
            else:
                print(
                    "⚠️  Batch API processing may not be effective; other methods used"
                )

            return True
        else:
            print(f"❌ Result count mismatch")
            return False

    except Exception as e:
        print(f"❌ Batch optimization test failed: {e}")
        return False


def main():
    print("🚀 Starting system tests (batch-optimized)...\n")

    tests = [
        ("Config load", test_config),
        ("GLM client (batch)", test_glm_client),
        ("Text processor", test_text_processor),
        ("File format support", test_file_formats),
        ("Error detector (batch)", test_error_detector),
        ("Batch optimization", test_batch_optimization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} exception: {e}")

    print(f"\n📊 Test results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Batch-optimized system is working.")
        print(
            "💡 Compared to the original, the new system should significantly reduce token usage and processing time."
        )
        return True
    else:
        print("⚠️  Some tests failed, please check configuration and network.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
