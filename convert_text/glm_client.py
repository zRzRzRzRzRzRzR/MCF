import json
import re
import time
from typing import Dict, List, Optional

import requests
from config import Config


class GLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GLM_API_KEY
        self.base_url = Config.GLM_BASE_URL
        self.model = Config.GLM_MODEL

        self.quick_fixes = {
            "我觉的": "我觉得",
            "你觉的": "你觉得",
            "他觉的": "他觉得",
            "她觉的": "她觉得",
            "说的好": "说得好",
            "做的不错": "做得不错",
            "想的周到": "想得周到",
            "听的清楚": "听得清楚",
            "跑的快": "跑得快",
            "写的好": "写得好",
            "学的认真": "学得认真",
            "睡的香": "睡得香",
            "工作的努力": "工作得努力",
            "来的及": "来得及",
            "记的": "记得",
            "舍的": "舍得",
            "值的": "值得",
            "在提醒": "再提醒",
            "在看看": "再看看",
            "在试试": "再试试",
            "在想想": "再想想",
            "在说说": "再说说",
            "在考虑": "再考虑",
            "在确认": "再确认",
            "在检查": "再检查",
            "在来": "再来",
            "因该": "应该",
            "发声": "发生",
            "做实": "做事",
            "那里": "哪里",
            "再那": "在那",
            "现再": "现在",
            "拔打": "拨打",
            "账本": "帐本",
            "申玉飞": "沈玉飞",
            "孙玉飞": "沈玉飞",
            "申一飞": "沈玉飞",
        }
        self.batch_size = 25  # 每批处理的文本数量
        self.max_tokens_per_request = 1200  # 每次请求的最大token数
        self.api_retry_limit = 2  # API重试次数

    def test_connection(self) -> bool:
        print("Testing API connectivity...")
        try:
            test_result = self._make_api_call("测试", max_tokens=10)
            if test_result:
                print("✅ API connection OK")
                return True
            else:
                print("❌ API connection failed, falling back to local processing")
                return False
        except Exception as e:
            print(f"❌ API connection exception: {e}")
            return False

    def _make_api_call(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
        }

        for attempt in range(self.api_retry_limit):
            try:
                response = requests.post(
                    f"{self.base_url}chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    response_json = response.json()
                    content = self._extract_content_safely(response_json)
                    if content:
                        return content
                elif response.status_code == 429:
                    time.sleep(2)  # 速率限制，等待后重试
                    continue
                else:
                    print(f"API error status code: {response.status_code}")
                    break

            except Exception as e:
                if attempt < self.api_retry_limit - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"API call failed: {e}")
                    break

        return None

    def _extract_content_safely(self, response_json: dict) -> str:
        if "choices" not in response_json or not response_json["choices"]:
            return ""

        choice = response_json["choices"][0]
        message = choice.get("message", {})

        # 优先使用content字段
        content = message.get("content", "").strip()

        # 如果content为空，尝试从reasoning_content提取
        if not content:
            reasoning = message.get("reasoning_content", "").strip()
            if reasoning:
                content = self._extract_answer_from_reasoning(reasoning)

        return content

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        lines = reasoning.split("\n")

        for line in lines:
            line = line.strip()
            # 查找包含修正结果的行
            if any(word in line for word in ["修正", "应该是", "正确的", "改为"]):
                # 提取引号中的内容
                quotes_match = re.findall(r'"([^"]*)"', line)
                if quotes_match:
                    return quotes_match[-1]

                # 提取冒号后的内容
                if "：" in line:
                    return line.split("：")[-1].strip()
                if ":" in line:
                    return line.split(":")[-1].strip()

        return ""

    def _apply_quick_fixes(self, text: str) -> tuple[str, list]:
        corrected_text = text
        errors = []

        for wrong, correct in self.quick_fixes.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                errors.append(
                    {
                        "type": "快速修正",
                        "original": wrong,
                        "corrected": correct,
                        "confidence": 0.95,
                    }
                )

        # 基本清理
        corrected_text = re.sub(r"\s+", " ", corrected_text).strip()
        corrected_text = re.sub(r"(.)\1{3,}", r"\1", corrected_text)  # 去重复字符

        return corrected_text, errors

    def _needs_api_processing(self, text: str) -> bool:
        text = text.strip()

        # 跳过太短的文本
        if len(text) < 8:
            return False

        # 跳过明显的标题行、时间戳行或系统信息
        skip_patterns = [
            r"^(发言人\d+|=+|-+|\d{2}:\d{2})",
            r"^chat-\d+",
            r"^\d{4}年",
            r"^文件|^转录|^记录",
        ]

        for pattern in skip_patterns:
            if re.match(pattern, text):
                return False

        # 检查是否包含可能需要修正的模式
        error_patterns = [
            r"[\u4e00-\u9fa5]+的[\u4e00-\u9fa5]+",  # 可能的"的/得"问题
            r"在[\u4e00-\u9fa5]{1,4}",  # 可能的"在/再"问题
            r"因[该当]",  # 应该相关
            r"[那哪]里",  # 哪里/那里混用
            r"[申孙][玉一][飞斐]",  # 人名变体
            r"[庞][加][莱来]",  # 庞加莱变体
        ]

        for pattern in error_patterns:
            if re.search(pattern, text):
                return True

        # 对于中长文本，如果包含常见词汇也考虑处理
        if len(text) > 15:
            common_words = ["我觉", "应该", "可以", "因为", "所以", "然后", "但是"]
            return any(word in text for word in common_words)

        return False

    def detect_and_correct_text_errors(
        self, text_segment: str, context: str = ""
    ) -> Dict:
        original_text = text_segment.strip()

        if not original_text:
            return self._create_result(original_text, original_text, False, [], "empty")

        # 第一层：快速修复
        quick_corrected, quick_errors = self._apply_quick_fixes(original_text)
        if quick_errors:
            return self._create_result(
                original_text, quick_corrected, True, quick_errors, "quick_fix"
            )

        # 第二层：预过滤
        if not self._needs_api_processing(original_text):
            return self._create_result(
                original_text, original_text, False, [], "pre_filter"
            )

        # 第三层：API处理
        try:
            prompt = self._create_optimized_prompt(original_text)
            api_response = self._make_api_call(prompt, max_tokens=150)

            if api_response and api_response.strip() != original_text:
                # 验证API响应的有效性
                cleaned_response = self._clean_api_response(api_response, original_text)
                if cleaned_response and cleaned_response != original_text:
                    return self._create_result(
                        original_text,
                        cleaned_response,
                        True,
                        [
                            {
                                "type": "API修正",
                                "original": original_text,
                                "corrected": cleaned_response,
                            }
                        ],
                        "api_correction",
                    )

        except Exception as e:
            print(f"API processing failed: {e}")

        # API失败或无修正，返回原文
        return self._create_result(original_text, original_text, False, [], "no_change")

    def _create_optimized_prompt(self, text: str) -> str:
        return f"""请修正以下文本中的错误，直接返回修正后的完整文本。如果没有错误，请返回原文。

原文：{text}

修正后："""

    def _clean_api_response(self, response: str, original_text: str) -> str:
        cleaned = response.strip()

        # 移除常见的标签和说明
        unwanted_patterns = [
            r"\*\*[^*]+\*\*",  # **标签**
            r"置信度[:：]\s*[\d.]+",  # 置信度信息
            r"修正[:：]",  # "修正："开头
            r"原文[:：]",  # "原文："开头
            r"错误详情[:：]",  # 错误详情
            r"\d+\.\s*",  # 数字编号
        ]

        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        cleaned = cleaned.strip()

        # 如果清理后的文本为空或明显不是有效文本，返回原文
        if not cleaned or len(cleaned) < len(original_text) * 0.3:
            return original_text

        return cleaned

    def _create_result(
        self, original: str, corrected: str, has_errors: bool, errors: list, method: str
    ) -> Dict:
        return {
            "original_text": original,
            "corrected_text": corrected,
            "has_errors": has_errors,
            "confidence": 0.9 if has_errors else 1.0,
            "errors": errors,
            "method": method,
        }

    def batch_detect_and_correct_segments(self, segments: List[Dict]) -> List[Dict]:
        print(f"Starting three-stage optimization for {len(segments)} segments...")

        results = []
        api_batch_texts = []
        api_batch_indices = []

        quick_fix_count = 0
        pre_filter_count = 0

        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()

            if not text:
                result = segment.copy()
                result.update(self._create_result(text, text, False, [], "empty"))
                results.append(result)
                continue

            quick_corrected, quick_errors = self._apply_quick_fixes(text)
            if quick_errors:
                result = segment.copy()
                result.update(
                    self._create_result(
                        text, quick_corrected, True, quick_errors, "quick_fix"
                    )
                )
                results.append(result)
                quick_fix_count += 1
                continue

            if not self._needs_api_processing(text):
                result = segment.copy()
                result.update(self._create_result(text, text, False, [], "pre_filter"))
                results.append(result)
                pre_filter_count += 1
                continue

            api_batch_texts.append(text)
            api_batch_indices.append(i)
            results.append(None)

        print(
            f"  Quick fixes: {quick_fix_count}, Pre-filter skipped: {pre_filter_count}"
        )

        if api_batch_texts:
            print(f"  Needs API processing: {len(api_batch_texts)} segments")
            api_results = self._batch_api_process(api_batch_texts)

            for idx, (original_idx, api_result) in enumerate(
                zip(api_batch_indices, api_results)
            ):
                segment = segments[original_idx]
                result = segment.copy()
                result.update(api_result)
                results[original_idx] = result

        total_corrections = sum(1 for r in results if r and r.get("has_errors", False))
        api_corrections = sum(
            1 for r in results if r and r.get("method") == "batch_api"
        )

        print(
            f"Batch processing completed! Total corrections: {total_corrections}, API corrections: {api_corrections}"
        )

        return results

    def _batch_api_process(self, texts: List[str]) -> List[Dict]:
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(
                f"    Processing batch {i // self.batch_size + 1}, segments: {len(batch)}"
            )

            try:
                batch_prompt = self._create_structured_batch_prompt(batch)
                api_response = self._make_api_call(
                    batch_prompt, max_tokens=self.max_tokens_per_request
                )

                if api_response:
                    batch_results = self._parse_structured_response(api_response, batch)
                    results.extend(batch_results)
                else:
                    for text in batch:
                        results.append(
                            self._create_result(text, text, False, [], "api_failed")
                        )

            except Exception as e:
                print(f"    Batch processing failed: {e}")
                for text in batch:
                    results.append(
                        self._create_result(text, text, False, [], "api_error")
                    )

            time.sleep(0.2)

        return results

    def _create_structured_batch_prompt(self, texts: List[str]) -> str:
        numbered_texts = []
        for i, text in enumerate(texts, 1):
            numbered_texts.append(f"{i}|{text}")

        prompt = f"""请修正以下文本中的错误，严格按照格式返回。每行一个结果，格式为"数字|修正后的文本"。如果某行没有错误，请原样返回该行。

输入：
{chr(10).join(numbered_texts)}

输出（仅返回修正后的文本，不要添加任何解释、标签或说明）："""

        return prompt

    def _parse_structured_response(
        self, response: str, original_texts: List[str]
    ) -> List[Dict]:
        results = []

        corrected_map = self._extract_corrections_from_response(
            response, original_texts
        )

        for i, original_text in enumerate(original_texts, 1):
            corrected_text = corrected_map.get(i, original_text)

            # 验证修正文本的有效性
            if corrected_text and corrected_text != original_text:
                # 清理修正文本
                cleaned_corrected = self._clean_api_response(
                    corrected_text, original_text
                )

                if cleaned_corrected and cleaned_corrected != original_text:
                    result = self._create_result(
                        original_text,
                        cleaned_corrected,
                        True,
                        [
                            {
                                "type": "批量API修正",
                                "original": original_text,
                                "corrected": cleaned_corrected,
                            }
                        ],
                        "batch_api",
                    )
                else:
                    result = self._create_result(
                        original_text, original_text, False, [], "batch_api_no_change"
                    )
            else:
                result = self._create_result(
                    original_text, original_text, False, [], "batch_api_no_change"
                )

            results.append(result)

        return results

    def _extract_corrections_from_response(
        self, response: str, original_texts: List[str]
    ) -> Dict[int, str]:
        corrected_map = {}
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 匹配 "数字|文本" 格式（主要格式）
            pipe_match = re.match(r"^(\d+)\|(.+)", line)
            if pipe_match:
                try:
                    num = int(pipe_match.group(1))
                    corrected = pipe_match.group(2).strip()
                    if corrected and num <= len(original_texts):
                        corrected_map[num] = corrected
                    continue
                except:
                    pass

            # 匹配 "数字. 文本" 格式（备用格式）
            dot_match = re.match(r"^(\d+)\.\s*(.+)", line)
            if dot_match:
                try:
                    num = int(dot_match.group(1))
                    corrected = dot_match.group(2).strip()
                    if corrected and num <= len(original_texts):
                        corrected_map[num] = corrected
                except:
                    pass

            # 匹配 "数字: 文本" 格式（备用格式）
            colon_match = re.match(r"^(\d+)[:：]\s*(.+)", line)
            if colon_match:
                try:
                    num = int(colon_match.group(1))
                    corrected = colon_match.group(2).strip()
                    if corrected and num <= len(original_texts):
                        corrected_map[num] = corrected
                except:
                    pass

        if not corrected_map and len(lines) == len(original_texts):
            for i, line in enumerate(lines, 1):
                cleaned_line = line.strip()
                if cleaned_line:
                    corrected_map[i] = cleaned_line

        return corrected_map

    def batch_detect_and_correct_texts(
        self, texts: List[str], batch_size: int = None
    ) -> List[Dict]:
        segments = []
        for i, text in enumerate(texts):
            segments.append(
                {
                    "line_number": i + 1,
                    "timestamp": "Unknown",
                    "speaker": "Unknown",
                    "text": text,
                    "original_line": text,
                }
            )

        results = self.batch_detect_and_correct_segments(segments)

        simple_results = []
        for result in results:
            simple_results.append(
                {
                    "original_text": result.get(
                        "original_text", result.get("text", "")
                    ),
                    "corrected_text": result.get(
                        "corrected_text", result.get("text", "")
                    ),
                    "has_errors": result.get("has_errors", False),
                    "confidence": result.get("confidence", 1.0),
                    "errors": result.get("errors", []),
                    "method": result.get("method", "unknown"),
                }
            )

        return simple_results

    def comprehensive_local_processing(self, text: str) -> Dict:
        original_text = text.strip()

        if len(original_text) < 2:
            return self._create_result(
                original_text, original_text, False, [], "too_short"
            )

        corrected_text, errors = self._apply_quick_fixes(original_text)
        has_errors = len(errors) > 0

        return {
            "original_text": original_text,
            "corrected_text": corrected_text,
            "has_errors": has_errors,
            "confidence": 0.95 if has_errors else 1.0,
            "errors": errors,
            "method": "local_processing",
        }
