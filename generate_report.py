from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, List, Tuple
from collections import defaultdict
import concurrent.futures
import os
from datetime import datetime
import re

class ReportGenerator:
    def __init__(self, llm_api,num_workers=4):
        self.llm_api = llm_api
        self.num_workers = num_workers

    def _generate_outline(self, main_question: str, dimensions: list) -> str:
        """生成报告大纲"""
        dimensions_str = "\n".join([f"- {dimension}" for dimension in dimensions])
        prompt = f"""
        ###目标###
        - 您是一位专业的研究报告撰写助手，需要根据 ### 用户问题 ### 生成一个技术报告的大纲。
        - 您可以参考 ### 维度 ### 生成章节标题
        - 您需要遵循 ### 格式要求 ### 中的说明。
        
        ### 格式要求 ###
        - 包含5-8个主要章节
        - 每个章节包含3-8个字
        - 需要包含“总结”章节
        - 末尾需有“参考文献”
        - 章节标题需体现专业性和逻辑性
        
        ### 用户问题 ###
        {main_question}
        
        ### 维度 ###
        {dimensions_str}

        ### 示例输出格式 ###
        # 某某漏洞威胁分析
         一. 漏洞概述
         二. 受影响版本
         三. 漏洞原理
           三(一) 漏洞实现方式
           三(二) 漏洞利用案例
         四. 结论
         参考文献
        """
        prompt = "no_think , " + prompt
        response = self.llm_api.invoke([SystemMessage(content=prompt)])
        return response.content.strip()

    def _refine_section(self, outline: str, text_chunk: str, main_question: str) -> str:
        """生成章节"""
        context = text_chunk
        prompt=f"""
        ### 目标 ###
            您是一位由 星图实验室 开发的大型语言人工智能助手。
            用户会向您提出问题，您将针对用户的问题，阅读文本块，并结合你已有的知识，完善以下报告章节内容：{outline}。
            您需要遵循 ### 报告格式 ### 中的说明。
                      
        ### 报告格式 ###
            1.文本块的内容每个都以[x]这样的编号开头，x代表一个数字。当你生成回答时，请按照引用编号[x]的格式在回答中对应部分引用文本块。请不要出现“根据信息[x]”等类似的语句，直接在引用的文本块后加上[x]。
            2.保持原有章节结构
            3.扩展专业案例分析（至少增加3个）
            4.如果文本块是英文，请仍然用中文生成答案。
            5.利用你已有的专业知识回答时，如果存在引用，也必须标出并加入参考文献。
            
        ### 风格 ###
            1. 绝不使用列表，而应将基于列表的信息转换为流畅的段落
            2. 仅对关键术语或发现保留粗体格式
            3. 在表格而非列表中呈现比较数据
            4. 参考文献可引用URL
            5. 使用主题句引导读者遵循逻辑进展
 
        ### 引用 ###
            - 您必须在每个直接使用文本块的句子后立即引用搜索结果。
            - 使用以下方法引用文本块。在相应句子末尾用方括号括起相关文本块的索引。例如："冰的密度小于水[1][2]。"
            - 每个索引应包含在自己的方括号中，绝不在单个方括号组中包含多个索引。
            - 最后一个词和引用之间不要留空格。
            - 如果文本块没有帮助，请用现有知识尽可能好地完善大纲。
            - 引用需汇总到参考文献，保留前15字即可。
 
        ### 受众 ###
            专业的信息安全从业者。
            
        ### 用户的问题 ###
            {main_question}
            
        ### 文本块内容 ###
            {context}"""
        prompt = "no_think , " + prompt
        response = self.llm_api.invoke([SystemMessage(content=prompt)])
        return response.content

    def _generate_report_from_knowledge(self, outline: str, question: str) -> str:
        """基于知识库生成完整报告"""
        prompt = f"""
        ### 任务要求 ###
        您是一位资深网络安全分析师，针对用户问题，请根据以下报告框架，基于您的专业知识完善技术报告内容：

        ### 报告框架 ###
        {outline}

        ### 用户问题 ###
        {question}

        ### 内容要求 ###
        1. 保持原有章节结构
        2. 为每个章节补充不少于200字的技术分析
        3. 包含实际漏洞案例
        4. 添加技术原理示意图描述（用文字说明）
        5. 在参考文献章节添加3-5个相关学术论文/标准文档


        ### 引用 ###
        1. 您必须在每个直接引用的句子后立即引用搜索结果。
        2. 在相应句子末尾用方括号括起索引。例如："冰的密度小于水[1][2]。"
        3. 每个索引应包含在自己的方括号中，绝不在单个方括号组中包含多个索引。
        4. 最后一个词和引用之间不要留空格。
        5.引用需汇总到参考文献，保留前15字即可，示例：
            参考文献
            [1]xxxxxxxxxx
            [2]xxxxxxxxxx
            [3]xxxxxxxxxx

        """
        prompt = "no_think , " + prompt
        messages = [SystemMessage(content=prompt)]
        response = self.llm_api.invoke(messages)
        return response.content

    def generate_report(
            self,
            chunks: list[str],
            user_question: str,
            dimensions:list
    ) -> str:
        """
        生成完整技术报告：
        - question_dimension_map: {query: dimension}
        """
        outline = self._generate_outline(user_question, dimensions)
        # 2. 分块处理文本内容
        if not chunks:
            return self._generate_report_from_knowledge(outline, user_question)

        final_report = outline
        for chunk in chunks:
            try:
                final_report = self._refine_section(
                    outline=final_report,
                    text_chunk=chunk,
                    main_question=user_question
                )
            except Exception as e:
                print(f"处理文本块时出错：{str(e)}")
                continue
        print(f"final_report: {final_report}")
        final_report = self._reorder_references(final_report)
        return final_report

    def _reorder_references(self, report_content: str) -> str:
        """重新排列报告中的引用序号和参考文献顺序"""
        # 1. 按出现顺序收集所有唯一引用
        in_text_refs = []
        seen = set()
        for match in re.finditer(r'\[(\d+)\]', report_content):
            ref_num = match.group(1)
            if ref_num not in seen:
                seen.add(ref_num)
                in_text_refs.append(ref_num)

        # 2. 创建新旧编号映射（按出现顺序）
        ref_map = {old: str(idx + 1) for idx, old in enumerate(in_text_refs)}

        # 3. 替换正文引用
        def replace_ref(match):
            old_num = match.group(1)
            return f"[{ref_map.get(old_num, old_num)}]"

        new_content = re.sub(r'\[(\d+)\]', replace_ref, report_content)

        # 4. 提取参考文献部分
        ref_match = re.search(
            r'(参考文献\s*\n+)(.*?)(?=\n{2,}|$)',
            new_content,
            flags=re.DOTALL | re.IGNORECASE
        )

        if not ref_match:
            return new_content  # 没有参考文献部分

        ref_title = ref_match.group(1)  # 保留"参考文献"标题
        ref_section = ref_match.group(2)

        # 5. 提取参考文献条目到字典
        ref_dict = {}
        entries = re.split(r'\n(?=\[\d+\])', ref_section)

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            # 匹配条目开头的编号和内容
            match = re.match(r'^\[(\d+)\]\s*(.*)', entry, re.DOTALL)
            if match:
                ref_num = match.group(1)
                content = match.group(2).replace('\n', ' ').strip()  # 将多行转换为单行
                ref_dict[ref_num] = content

        # 6. 仅处理被引用的条目（按正文出现顺序），并截断内容
        new_ref_entries = []
        for old_num in in_text_refs:
            if old_num in ref_dict:
                new_num = ref_map[old_num]
                # 截取前20个字符（中文字符安全）
                content = ref_dict[old_num]
                truncated_content = content[:20] + ('...' if len(content) > 20 else '')
                new_ref_entries.append(f"[{new_num}] {truncated_content}")

        # 7. 重建参考文献部分
        new_ref_section = "\n".join(new_ref_entries)
        return new_content.replace(
            ref_match.group(0),
            ref_title + new_ref_section
        )


    def save_report_to_txt(
                self,
                report_content: str,
                save_dir: str = "report"  # 修改默认值为None
        ) -> str:
            """
            保存报告到当前目录（或指定目录）
            """
            # 获取当前Python文件所在目录
            try:
                # 确定存储目录
                if save_dir:
                    base_dir = os.path.abspath(save_dir)
                else:
                    base_dir = os.getcwd()  # 使用当前工作目录

                # 确保目录存在
                os.makedirs(base_dir, exist_ok=True)

                # 生成安全文件名（关键修复）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 从报告内容提取更可靠的文件名
                clean_title = "技术分析报告"
                filename = f"{timestamp}_{clean_title}.txt"
                filepath = os.path.join(base_dir, filename)

                with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(report_content)

                return os.path.abspath(filepath)
            except Exception as e:
                raise RuntimeError(f"文件保存失败：{str(e)}\n尝试路径：{filepath}")
