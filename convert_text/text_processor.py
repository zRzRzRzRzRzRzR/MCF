import re
import jieba
from typing import List, Dict

class TextProcessor:
    def __init__(self):
        jieba.initialize()

    def parse_transcription_file(self, file_path: str) -> List[Dict]:
        """
        解析转录文件，自动识别格式并提取时间戳、发言人和文本
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        file_format = self._detect_format(content)
        print(f"🔍 Detected transcription file format: {file_format}")

        if file_format == 'speaker_timestamp':
            return self._parse_speaker_timestamp_format(content)
        elif file_format == 'timestamp_speaker':
            return self._parse_timestamp_speaker_format(content)
        else:
            return self._parse_mixed_format(content)

    def _detect_format(self, content: str) -> str:
        """
        自动检测转录文件的格式
        """
        lines = content.strip().split('\n')

        speaker_timestamp_count = 0
        timestamp_speaker_count = 0

        for line in lines[:20]:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^发言人\d+\s+\d{2}:\d{2}', line):
                speaker_timestamp_count += 1

            if re.match(r'\[\d{2}:\d{2}:\d{2}', line):
                timestamp_speaker_count += 1

        if speaker_timestamp_count > timestamp_speaker_count:
            return 'speaker_timestamp'
        elif timestamp_speaker_count > 0:
            return 'timestamp_speaker'
        else:
            return 'mixed'

    def _parse_speaker_timestamp_format(self, content: str) -> List[Dict]:
        """
        解析 "发言人X 时间戳" 后跟内容的格式
        """
        lines = content.strip().split('\n')
        segments = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line or self._is_header_line(line):
                i += 1
                continue

            pattern = r'^发言人(\d+)\s+(\d{2}:\d{2})'
            match = re.match(pattern, line)

            if match:
                speaker_id = match.group(1)
                timestamp = match.group(2)
                speaker = f"发言人{speaker_id}"

                content_lines = []
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()

                    if not next_line:
                        j += 1
                        continue

                    if re.match(r'^发言人\d+\s+\d{2}:\d{2}', next_line):
                        break

                    content_lines.append(next_line)
                    j += 1

                if content_lines:
                    full_text = '\n'.join(content_lines)
                    segments.append({
                        'line_number': i + 1,
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'text': full_text,
                        'original_line': line
                    })

                i = j
            else:
                i += 1

        return segments

    def _parse_timestamp_speaker_format(self, content: str) -> List[Dict]:
        """
        解析传统的时间戳+发言人格式
        """
        lines = content.strip().split('\n')
        segments = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or self._is_header_line(line):
                continue

            segment = self._parse_traditional_line(line, line_num)
            if segment:
                segments.append(segment)

        return segments

    def _parse_mixed_format(self, content: str) -> List[Dict]:
        """
        解析混合格式或纯文本格式
        """
        lines = content.strip().split('\n')
        segments = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or self._is_header_line(line):
                continue

            segment = self._parse_traditional_line(line, line_num)

            if not segment:
                segment = {
                    'line_number': line_num,
                    'timestamp': 'Unknown',
                    'speaker': 'Unknown',
                    'text': line,
                    'original_line': line
                }

            segments.append(segment)

        return segments

    def _parse_traditional_line(self, line: str, line_num: int) -> Dict:
        """
        解析传统格式的单行文本
        """
        pattern1 = r'\[(\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.+)'
        match1 = re.match(pattern1, line)
        if match1:
            return {
                'line_number': line_num,
                'timestamp': match1.group(1),
                'speaker': match1.group(2).strip(),
                'text': match1.group(3).strip(),
                'original_line': line
            }

        pattern2 = r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)'
        match2 = re.match(pattern2, line)
        if match2:
            return {
                'line_number': line_num,
                'timestamp': match2.group(1),
                'speaker': 'Unknown',
                'text': match2.group(2).strip(),
                'original_line': line
            }

        pattern3 = r'([^[]+)\[(\d{2}:\d{2}:\d{2})\]:\s*(.+)'
        match3 = re.match(pattern3, line)
        if match3:
            return {
                'line_number': line_num,
                'timestamp': match3.group(2),
                'speaker': match3.group(1).strip(),
                'text': match3.group(3).strip(),
                'original_line': line
            }

        pattern4 = r'^(发言人\d+|[\u4e00-\u9fa5]{2,8})[:：]\s*(.+)'
        match4 = re.match(pattern4, line)
        if match4:
            return {
                'line_number': line_num,
                'timestamp': 'Unknown',
                'speaker': match4.group(1),
                'text': match4.group(2).strip(),
                'original_line': line
            }

        return None

    def _is_header_line(self, line: str) -> bool:
        """
        判断是否是标题行或无关行
        """
        header_patterns = [
            r'^chat-\d+',
            r'^\d{4}年',
            r'^=+$',
            r'^-+$',
            r'^文件|^转录|^记录',
            r'^$'
        ]

        for pattern in header_patterns:
            if re.match(pattern, line):
                return True
        return False

    def segment_long_text(self, text: str, max_length: int = 200) -> List[str]:
        """
        将长文本分割成较短的段落，便于API处理
        """
        if len(text) <= max_length:
            return [text]

        sentences = re.split(r'[。！？\n]', text)
        segments = []
        current_segment = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_segment = current_segment + sentence + "。" if current_segment else sentence

            if len(test_segment) <= max_length:
                current_segment = test_segment
            else:
                if current_segment:
                    segments.append(current_segment)

                if len(sentence) > max_length:
                    while sentence:
                        segments.append(sentence[:max_length])
                        sentence = sentence[max_length:]
                    current_segment = ""
                else:
                    current_segment = sentence + "。"

        if current_segment:
            segments.append(current_segment)

        return segments

    def clean_text(self, text: str) -> str:
        """
        清理文本，移除多余的空白和特殊字符
        """
        text = re.sub(r'\s+', ' ', text)

        text = text.strip()

        text = re.sub(r'[嗯啊呃哎]{2,}', '嗯', text)
        
        return text