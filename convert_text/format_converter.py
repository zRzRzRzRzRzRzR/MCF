#!/usr/bin/env python3
"""
将chat-2402格式转换为chat-1533格式的脚本
"""

import re
import os
from datetime import datetime

def convert_format(input_file: str, output_file: str = None):
    """
    将chat-2402格式转换为chat-1533格式
    """
    if not output_file:
        # 自动生成输出文件名
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.txt"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    converted_lines = []
    i = 0
    
    # 添加标题行（模仿chat-1533格式）
    file_number = extract_file_number(input_file)
    converted_lines.append(f"{file_number}_原文")
    
    # 提取日期信息
    date_line = extract_date_from_content(content)
    if date_line:
        converted_lines.append(f"               {date_line}")
    else:
        converted_lines.append(f"               {datetime.now().strftime('%Y 年 %m 月 %d 日 %H:%M')}")
    
    # 处理内容行
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过标题和无关行
        if should_skip_line(line):
            i += 1
            continue
        
        # 检查是否是发言人行
        speaker_match = re.match(r'^发言人\s*(\d+)\s+(\d{2}:\d{2})', line)
        if speaker_match:
            speaker_id = speaker_match.group(1)
            timestamp = speaker_match.group(2)
            
            # 收集该发言人的所有内容
            content_lines = []
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j].strip()
                
                # 跳过空行
                if not next_line:
                    j += 1
                    continue
                
                # 如果遇到下一个发言人，停止
                if re.match(r'^发言人\s*\d+\s+\d{2}:\d{2}', next_line):
                    break
                
                # 跳过"Unknown Unknown"等无关行
                if should_skip_line(next_line):
                    j += 1
                    continue
                
                # 收集内容行
                content_lines.append(next_line)
                j += 1
            
            # 如果有内容，添加到转换结果中
            if content_lines:
                # 添加发言人行
                converted_lines.append(f"发言人 {speaker_id} {timestamp}")
                
                # 添加内容（合并多行或保持分行）
                if len(content_lines) == 1:
                    # 单行内容
                    converted_lines.append(content_lines[0])
                else:
                    # 多行内容，保持原有换行
                    for content_line in content_lines:
                        converted_lines.append(content_line)
            
            i = j
        else:
            i += 1
    
    # 写入转换后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines))
    
    print(f"格式转换完成：{input_file} -> {output_file}")
    return output_file

def extract_file_number(filename: str) -> str:
    """从文件名中提取编号"""
    # 尝试匹配chat-数字格式
    match = re.search(r'chat[_-]?(\d+)', filename.lower())
    if match:
        return match.group(1)
    
    # 如果没找到，返回默认值
    return "unknown"

def extract_date_from_content(content: str) -> str:
    """从内容中提取日期信息"""
    # 查找修正时间行
    match = re.search(r'修正时间:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
    if match:
        date_str = match.group(1)
        try:
            # 转换为目标格式
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y 年 %m 月 %d 日 %H:%M')
        except:
            pass
    
    # 查找其他日期格式
    match = re.search(r'(\d{4})\s*年\s*(\d{2})\s*月\s*(\d{2})\s*日\s*(\d{2}):(\d{2})', content)
    if match:
        return f"{match.group(1)} 年 {match.group(2)} 月 {match.group(3)} 日 {match.group(4)}:{match.group(5)}"
    
    return None

def should_skip_line(line: str) -> bool:
    """判断是否应该跳过该行"""
    if not line:
        return True
    
    skip_patterns = [
        r'^Unknown\s*Unknown$',
        r'^Unknown$',
        r'^chat[_-]?\d+.*自动修正版',
        r'^修正时间:',
        r'^原始文件:',
        r'^=+$',
        r'^-+$',
        r'^\d{4}\s*年.*\d{2}:\d{2}$',  # 单独的日期时间行
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, line):
            return True
    
    return False

def batch_convert(input_pattern: str):
    """批量转换文件"""
    import glob
    
    files = glob.glob(input_pattern)
    if not files:
        print(f"未找到匹配的文件: {input_pattern}")
        return
    
    print(f"找到 {len(files)} 个文件待转换")
    
    for file in files:
        try:
            convert_format(file)
        except Exception as e:
            print(f"转换失败 {file}: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='转换chat文件格式')
    parser.add_argument('input', help='输入文件路径或通配符')
    parser.add_argument('-o', '--output', help='输出文件路径（可选）')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert(args.input)
    else:
        if os.path.isfile(args.input):
            convert_format(args.input, args.output)
        else:
            print(f"文件不存在: {args.input}")

if __name__ == "__main__":
    main()