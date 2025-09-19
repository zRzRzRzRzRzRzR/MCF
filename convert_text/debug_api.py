# 在你现有的 glm_client.py 中添加以下修复代码

import re
import requests
import json
from typing import Dict, List

class GLMClient:
    def __init__(self, api_key: str = None):
        # 保持你现有的初始化代码...
        self.comprehensive_fixes = {
            '我觉的': '我觉得', '你觉的': '你觉得', '他觉的': '他觉得', '她觉的': '她觉得',
            '说的好': '说得好', '做的不错': '做得不错', '想的周到': '想得周到',
            '听的清楚': '听得清楚', '跑的快': '跑得快', '写的好': '写得好',
            '学的认真': '学得认真', '睡的香': '睡得香', '工作的努力': '工作得努力',
            '讲的': '讲得', '唱的': '唱得', '演的': '演得', '看的': '看得',
            '玩的开心': '玩得开心', '过的': '过得', '活的': '活得',
            '来的及': '来得及', '记的': '记得', '舍的': '舍得', '值的': '值得',
            '在提醒': '再提醒', '在看看': '再看看', '在试试': '再试试',
            '在想想': '再想想', '在说说': '再说说', '在考虑': '再考虑',
            '在确认': '再确认', '在检查': '再检查', '在来': '再来',
            '在做做': '再做做', '在等等': '再等等', '在问问': '再问问',
            '因该': '应该', '发声': '发生', '做实': '做事',
            '那里': '哪里', '再那': '在那', '现再': '现在',
            '拔打': '拨打', '账本': '帐本', '旁将来': '庞加莱',
            '申玉飞': '沈玉飞', '孙玉飞': '沈玉飞', '申一飞': '沈玉飞'
        }

    def test_connection(self) -> bool:
        """修复后的连接测试"""
        print("测试API连接...")
        try:
            # 使用简单的测试请求
            response = self._make_safe_api_request("你好", max_tokens=50)
            if response and len(response.strip()) > 0:
                print("✅ API连接正常")
                return True
            else:
                print("❌ API返回空响应，将使用本地处理模式")
                return False
        except Exception as e:
            print(f"❌ API连接失败: {e}，将使用本地处理模式")
            return False

    def _make_safe_api_request(self, prompt: str, max_tokens: int = 300) -> str:
        """
        修复GLM-4.5空响应问题的安全API请求方法
        """
        from config import Config
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {Config.GLM_API_KEY}'
        }
        
        # 优化prompt避免触发推理模式
        optimized_prompt = self._optimize_prompt_for_glm45(prompt)
        
        payload = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": optimized_prompt}],
            "max_tokens": max_tokens,  # 增加token限制
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                f'{Config.GLM_BASE_URL}chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                return None
            
            response_json = response.json()
            
            # 修复：正确提取响应内容
            content = self._extract_content_safely(response_json)
            
            return content if content else None
            
        except Exception as e:
            print(f"API请求异常: {e}")
            return None

    def _optimize_prompt_for_glm45(self, prompt: str) -> str:
        """
        优化prompt避免触发GLM-4.5的推理模式
        """
        # 如果是纠错请求，使用更直接的格式
        if any(word in prompt for word in ['修正', '纠错', '检查', '改善', '优化']):
            # 提取要修正的文本
            if '：' in prompt:
                text_to_fix = prompt.split('：')[-1].strip()
            else:
                text_to_fix = prompt
            
            # 使用更直接的指令格式
            return f"请直接输出修正后的文本：{text_to_fix}"
        
        return prompt

    def _extract_content_safely(self, response_json: dict) -> str:
        """
        安全提取GLM-4.5响应内容，处理空content字段问题
        """
        if 'choices' not in response_json or not response_json['choices']:
            return ""
        
        choice = response_json['choices'][0]
        message = choice.get('message', {})
        
        # 优先使用content字段
        content = message.get('content', '').strip()
        
        # 如果content为空，尝试从reasoning_content提取
        if not content:
            reasoning = message.get('reasoning_content', '').strip()
            if reasoning:
                # 从推理内容中提取实际答案
                extracted = self._extract_answer_from_reasoning(reasoning)
                if extracted:
                    content = extracted
        
        return content

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        """
        从reasoning_content中提取实际的修正结果
        """
        # 查找推理中的关键修正信息
        lines = reasoning.split('\n')
        
        for line in lines:
            line = line.strip()
            # 查找包含修正结果的行
            if any(word in line for word in ['修正', '应该是', '正确的', '改为']):
                # 提取引号中的内容
                import re
                quotes_match = re.findall(r'"([^"]*)"', line)
                if quotes_match:
                    return quotes_match[-1]  # 返回最后一个引号内容
                    
                # 提取冒号后的内容
                if '：' in line:
                    return line.split('：')[-1].strip()
                if ':' in line:
                    return line.split(':')[-1].strip()
        
        return ""

    def comprehensive_local_processing(self, text: str) -> Dict:
        """
        增强的本地处理方法（现有代码的改进版）
        """
        original_text = text.strip()
        
        if len(original_text) < 2:
            return self._create_clean_result(original_text, 'too_short')
        
        corrected_text = original_text
        errors = []
        
        # 应用本地修正规则
        for wrong, correct in self.comprehensive_fixes.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                errors.append({
                    'type': '自动修正',
                    'original': wrong,
                    'corrected': correct,
                    'confidence': 0.95
                })
        
        # 基本清理
        corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()
        corrected_text = re.sub(r'(.)\1{2,}', r'\1', corrected_text)
        corrected_text = re.sub(r'[！]{2,}', '！', corrected_text)
        corrected_text = re.sub(r'[？]{2,}', '？', corrected_text)
        
        has_errors = len(errors) > 0 or corrected_text != original_text
        
        return {
            'original_text': original_text,
            'corrected_text': corrected_text,
            'has_errors': has_errors,
            'confidence': 0.93 if has_errors else 1.0,
            'errors': errors,
            'method': 'comprehensive_local'
        }

    def detect_and_correct_text_errors(self, text_segment: str, context: str = "") -> Dict:
        """
        修复后的错误检测方法，混合本地和API处理
        """
        # 首先尝试本地处理
        local_result = self.comprehensive_local_processing(text_segment)
        
        # 如果本地处理发现了错误，直接返回
        if local_result['has_errors']:
            return local_result
        
        # 如果本地没发现问题，尝试API处理（但要处理空响应问题）
        try:
            api_response = self._make_safe_api_request(
                f"修正文本错误：{text_segment}", 
                max_tokens=200
            )
            
            if api_response and api_response.strip() != text_segment.strip():
                return {
                    'original_text': text_segment,
                    'corrected_text': api_response.strip(),
                    'has_errors': True,
                    'confidence': 0.85,
                    'errors': [{'type': 'API修正', 'original': text_segment, 'corrected': api_response}],
                    'method': 'api_correction'
                }
        except Exception as e:
            print(f"API处理失败: {e}")
        
        # API失败或无修正，返回本地结果
        return local_result

    def batch_detect_and_correct_segments(self, segments: List[Dict], max_retries: int = 2) -> List[Dict]:
        """
        修复后的批量处理方法
        """
        print(f"启动修复版本处理模式，处理 {len(segments)} 个段落...")
        
        results = []
        total_corrections = 0
        api_successes = 0
        local_corrections = 0
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            
            if not text:
                result = segment.copy()
                result.update(self._create_clean_result(text, 'empty'))
                results.append(result)
                continue
            
            try:
                process_result = self.detect_and_correct_text_errors(text)
                
                result = segment.copy()
                result.update(process_result)
                results.append(result)
                
                if process_result['has_errors']:
                    total_corrections += 1
                    if process_result['method'] == 'api_correction':
                        api_successes += 1
                    else:
                        local_corrections += 1
                
            except Exception as e:
                print(f"处理段落{i+1}时出错: {e}")
                result = segment.copy()
                result.update({
                    'error': str(e),
                    'original_text': text,
                    'corrected_text': text,
                    'has_errors': False,
                    'method': 'error'
                })
                results.append(result)
            
            if (i + 1) % 50 == 0 or i == len(segments) - 1:
                print(f"  进度: {i + 1}/{len(segments)} ({(i+1)/len(segments)*100:.1f}%)")
        
        print(f"处理完成！总修正: {total_corrections}, API修正: {api_successes}, 本地修正: {local_corrections}")
        return results

    def _create_clean_result(self, text: str, method: str = 'local_only') -> Dict:
        return {
            'original_text': text,
            'corrected_text': text,
            'has_errors': False,
            'confidence': 1.0,
            'errors': [],
            'method': method
        }

    # 保持现有的其他方法...
    def batch_detect_and_correct_texts(self, texts: List[str], batch_size: int = None) -> List[Dict]:
        segments = []
        for i, text in enumerate(texts):
            segments.append({
                'line_number': i + 1,
                'timestamp': 'Unknown',
                'speaker': 'Unknown',
                'text': text,
                'original_line': text
            })
        
        results = self.batch_detect_and_correct_segments(segments)
        
        simple_results = []
        for result in results:
            simple_results.append({
                'original_text': result.get('original_text', result.get('text', '')),
                'corrected_text': result.get('corrected_text', result.get('text', '')),
                'has_errors': result.get('has_errors', False),
                'confidence': result.get('confidence', 1.0),
                'errors': result.get('errors', []),
                'method': result.get('method', 'local_only')
            })
        

        return simple_results
