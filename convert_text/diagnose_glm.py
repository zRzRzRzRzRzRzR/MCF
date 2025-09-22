import json
import time

import requests
from config import Config


def test_basic_api_call():
    print("🔍 Testing basic API call...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.GLM_API_KEY}",
    }

    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 50,
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            f"{Config.GLM_BASE_URL}chat/completions",
            headers=headers,
            json=payload,
            timeout=10,
        )

        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response length: {len(response.text)}")
        print(f"Response preview: {response.text[:500]}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                print(
                    f"JSON structure: {json.dumps(response_json, ensure_ascii=False, indent=2)[:500]}"
                )
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse failed: {e}")
                return False
        else:
            print("❌ Unexpected HTTP status code")
            return False

    except Exception as e:
        print(f"❌ Request exception: {e}")
        return False


def test_different_models():
    print("\n🔍 Testing different models...")

    models = ["glm-4.5", "glm-4", "glm-4-0520", "glm-4-plus"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.GLM_API_KEY}",
    }

    for model in models:
        print(f"\nTesting Model: {model}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 20,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                f"{Config.GLM_BASE_URL}chat/completions",
                headers=headers,
                json=payload,
                timeout=10,
            )

            print(f"  Status code: {response.status_code}")
            print(f"  Response length: {len(response.text)}")

            if len(response.text) > 0:
                print(f"  ✅ {model} works")
                return model
            else:
                print(f"  ❌ {model} returned empty response")

        except Exception as e:
            print(f"  ❌ {model} request failed: {e}")

    return None


def test_content_filtering():
    """Test potential content filtering behavior"""
    print("\n🔍 Testing content filtering...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.GLM_API_KEY}",
    }

    test_contents = [
        "你好",
        "今天天气不错",
        "我觉的这个方案不错",  # 包含错误的文本
        "修正以下文本中的错误：我觉的很好",  # 修正任务
        "请检查这段话：在说一遍",  # 另一个修正任务
    ]

    for i, content in enumerate(test_contents):
        print(f"\nTest item {i + 1}: {content}")

        payload = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                f"{Config.GLM_BASE_URL}chat/completions",
                headers=headers,
                json=payload,
                timeout=10,
            )

            print(
                f"  Status code: {response.status_code}, length: {len(response.text)}"
            )

            if len(response.text) > 0:
                print("  ✅ Response OK")
            else:
                print("  ❌ Possibly filtered")

        except Exception as e:
            print(f"  ❌ Request failed: {e}")


def test_api_quota():
    """Test API quota and rate limiting"""
    print("\n🔍 Testing API quota...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.GLM_API_KEY}",
    }

    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "测试"}],
        "max_tokens": 10,
        "temperature": 0.1,
    }

    # Burst requests
    for i in range(5):
        print(f"  Burst request {i + 1}/5...")

        try:
            response = requests.post(
                f"{Config.GLM_BASE_URL}chat/completions",
                headers=headers,
                json=payload,
                timeout=10,
            )

            print(
                f"    Status code: {response.status_code}, length: {len(response.text)}"
            )

            if response.status_code == 429:
                print("    ⚠️ Rate limit triggered")
                break
            elif response.status_code != 200:
                print(f"    ❌ Other error: {response.status_code}")
                break

        except Exception as e:
            print(f"    ❌ Request exception: {e}")

        time.sleep(0.5)


def test_auth_and_key():
    """Test auth and API key behavior"""
    print("\n🔍 Testing authentication...")

    # Test invalid key
    invalid_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_key_test",
    }

    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 10,
    }

    try:
        response = requests.post(
            f"{Config.GLM_BASE_URL}chat/completions",
            headers=invalid_headers,
            json=payload,
            timeout=10,
        )

        print(f"Invalid key test - status code: {response.status_code}")
        print(f"Invalid key test - response: {response.text[:200]}")

    except Exception as e:
        print(f"Invalid key test exception: {e}")

    # Test current key format
    print("\nCurrent key format check:")
    print(f"Key length: {len(Config.GLM_API_KEY)}")
    print(f"Key prefix: {Config.GLM_API_KEY[:10]}...")
    print(f"Key suffix: ...{Config.GLM_API_KEY[-10:]}")


def test_simplified_correction():
    """Test simplified correction prompts"""
    print("\n🔍 Testing simplified correction...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.GLM_API_KEY}",
    }

    # Minimal correction prompts
    simple_prompts = ["修正：我觉的很好", "纠错：在说一遍", "检查：因该好好学习"]

    for prompt in simple_prompts:
        print(f"\nTest prompt: {prompt}")

        payload = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                f"{Config.GLM_BASE_URL}chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )

            print(f"  Status code: {response.status_code}")
            print(f"  Response length: {len(response.text)}")

            if len(response.text) > 0:
                try:
                    response_json = response.json()
                    if "choices" in response_json and response_json["choices"]:
                        content = response_json["choices"][0]["message"]["content"]
                        print(f"  ✅ Response: {content}")
                    else:
                        print("  ❌ Unexpected response format")
                except:
                    print("  ❌ JSON parse failed")
            else:
                print("  ❌ Empty response")

        except Exception as e:
            print(f"  ❌ Request exception: {e}")


def main():
    """Main diagnostic flow"""
    print("🚀 GLM API deep diagnostics starting...\n")

    print("Configuration:")
    print(f"API Base URL: {Config.GLM_BASE_URL}")
    print(f"API Key: {Config.GLM_API_KEY[:8]}...{Config.GLM_API_KEY[-4:]}")
    print("Default model: glm-4.5")
    print("=" * 60)

    tests = [
        ("Basic API call", test_basic_api_call),
        ("Different models test", test_different_models),
        ("Content filtering test", test_content_filtering),
        ("API quota test", test_api_quota),
        ("Auth test", test_auth_and_key),
        ("Simplified correction test", test_simplified_correction),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} execution error: {e}")
            results[test_name] = False

        time.sleep(2)  # Avoid too frequent requests

    # Summary of diagnostic results
    print(f"\n{'=' * 60}")
    print("📊 Diagnostic summary")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ OK" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

    # Suggestions
    print("\n💡 Suggestions:")
    if not results.get("Basic API call", False):
        print("- Basic API call failed. Check network connectivity and API key.")

    working_model = results.get("Different models test")
    if working_model:
        print(f"- Suggested working model: {working_model}")

    if not results.get("Content filtering test", False):
        print("- Possible content filtering. Try simpler prompts.")

    if not results.get("API quota test", False):
        print("- Possible rate limiting. Increase delay between requests.")


if __name__ == "__main__":
    main()
