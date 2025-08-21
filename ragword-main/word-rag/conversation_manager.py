import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """对话记忆管理"""
    
    def __init__(self, short_term_window: int = 5, long_term_threshold: int = 10):
        self.short_term_window = short_term_window
        self.long_term_threshold = long_term_threshold
        
        # 短期记忆（最近N轮对话）
        self.short_term_memory = []
        
        # 长期记忆（向量化存储）
        self.long_term_memory = None
        self.embeddings = None
        
        # 对话摘要
        self.conversation_summaries = []
    
    def add_interaction(self, user_query: str, system_response: str, context_docs: List[Document] = None):
        """添加一轮对话交互"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'system_response': system_response,
            'context_documents': len(context_docs) if context_docs else 0,
            'context_summary': self._summarize_context(context_docs) if context_docs else ""
        }
        
        # 添加到短期记忆
        self.short_term_memory.append(interaction)
        
        # 保持短期记忆窗口大小
        if len(self.short_term_memory) > self.short_term_window:
            self.short_term_memory.pop(0)
        
        # 检查是否需要转移到长期记忆
        if len(self.short_term_memory) >= self.long_term_threshold:
            self._transfer_to_long_term()
    
    def _summarize_context(self, documents: List[Document]) -> str:
        """摘要上下文文档"""
        if not documents:
            return ""
        
        # 简单的摘要：提取关键信息
        summaries = []
        for i, doc in enumerate(documents[:3]):  # 只摘要前3个文档
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            summaries.append(f"文档{i+1}: {content}")
        
        return "; ".join(summaries)
    
    def _transfer_to_long_term(self):
        """将短期记忆转移到长期记忆"""
        if not self.short_term_memory:
            return
        
        # 创建对话摘要
        summary = self._create_conversation_summary()
        self.conversation_summaries.append(summary)
        
        # 清空短期记忆
        self.short_term_memory = []
        
        logger.info("短期记忆已转移到长期记忆")
    
    def _create_conversation_summary(self) -> Dict[str, Any]:
        """创建对话摘要"""
        if not self.short_term_memory:
            return {}
        
        # 提取关键信息
        queries = [interaction['user_query'] for interaction in self.short_term_memory]
        responses = [interaction['system_response'] for interaction in self.short_term_memory]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'interaction_count': len(self.short_term_memory),
            'key_topics': self._extract_key_topics(queries),
            'main_questions': queries,
            'response_summary': self._summarize_responses(responses)
        }
        
        return summary
    
    def _extract_key_topics(self, queries: List[str]) -> List[str]:
        """提取关键主题"""
        # 简单的关键词提取
        topics = set()
        for query in queries:
            # 提取常见问题词
            if '什么是' in query:
                topics.add('定义解释')
            elif '如何' in query or '怎么' in query:
                topics.add('方法步骤')
            elif '为什么' in query:
                topics.add('原因分析')
            elif '比较' in query or '区别' in query:
                topics.add('对比分析')
        
        return list(topics)
    
    def _summarize_responses(self, responses: List[str]) -> str:
        """摘要响应内容"""
        if not responses:
            return ""
        
        # 简单的摘要：取第一个响应的前100字符
        return responses[0][:100] + "..." if len(responses[0]) > 100 else responses[0]
    
    def get_recent_context(self, n: int = None) -> List[Dict[str, Any]]:
        """获取最近的对话上下文"""
        n = n or self.short_term_window
        return self.short_term_memory[-n:] if self.short_term_memory else []
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取完整对话历史"""
        return self.conversation_summaries + self.short_term_memory

class DialogueStateManager:
    """对话状态管理"""
    
    def __init__(self):
        self.current_state = {
            'active_documents': set(),  # 当前活跃的文档
            'focus_topics': set(),      # 当前关注的主题
            'user_preferences': {},     # 用户偏好
            'conversation_mode': 'general'  # 对话模式
        }
        
        self.slot_filling = {
            'document_name': None,
            'chapter': None,
            'topic': None,
            'time_range': None
        }
    
    def update_state(self, user_input: str, context_docs: List[Document] = None):
        """更新对话状态"""
        # 解析用户输入中的槽位信息
        self._extract_slots(user_input)
        
        # 更新活跃文档
        if context_docs:
            doc_names = set()
            for doc in context_docs:
                if 'source' in doc.metadata:
                    doc_names.add(doc.metadata['source'])
            self.current_state['active_documents'].update(doc_names)
        
        # 更新关注主题
        topics = self._extract_topics(user_input)
        self.current_state['focus_topics'].update(topics)
        
        # 检测对话模式
        self._detect_conversation_mode(user_input)
    
    def _extract_slots(self, user_input: str):
        """提取槽位信息"""
        # 文档名提取
        if '文档' in user_input or '文件' in user_input:
            # 简单的文档名提取逻辑
            import re
            doc_match = re.search(r'文档[：:]\s*([^\s，。]+)', user_input)
            if doc_match:
                self.slot_filling['document_name'] = doc_match.group(1)
        
        # 章节提取
        if '章' in user_input or '节' in user_input:
            chapter_match = re.search(r'第?(\d+)章|第?(\d+)节', user_input)
            if chapter_match:
                chapter_num = chapter_match.group(1) or chapter_match.group(2)
                self.slot_filling['chapter'] = f"第{chapter_num}章"
        
        # 主题提取
        topic_keywords = ['关于', '主题', '内容']
        for keyword in topic_keywords:
            if keyword in user_input:
                # 提取主题词
                words = user_input.split(keyword)
                if len(words) > 1:
                    topic = words[1].split('，')[0].split('。')[0].strip()
                    if topic:
                        self.slot_filling['topic'] = topic
                break
    
    def _extract_topics(self, user_input: str) -> List[str]:
        """提取主题"""
        topics = []
        
        # 基于关键词的主题分类
        topic_mapping = {
            '定义': ['什么是', '定义', '概念', '含义'],
            '方法': ['如何', '怎么', '方法', '步骤', '流程'],
            '原因': ['为什么', '原因', '理由', '动机'],
            '比较': ['比较', '区别', '差异', '对比'],
            '应用': ['应用', '使用', '实践', '案例']
        }
        
        for topic, keywords in topic_mapping.items():
            if any(keyword in user_input for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _detect_conversation_mode(self, user_input: str):
        """检测对话模式"""
        if '只' in user_input or '仅' in user_input:
            self.current_state['conversation_mode'] = 'focused'
        elif '所有' in user_input or '全部' in user_input:
            self.current_state['conversation_mode'] = 'comprehensive'
        else:
            self.current_state['conversation_mode'] = 'general'
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'state': self.current_state.copy(),
            'slots': self.slot_filling.copy()
        }
    
    def reset_state(self):
        """重置状态"""
        self.current_state = {
            'active_documents': set(),
            'focus_topics': set(),
            'user_preferences': {},
            'conversation_mode': 'general'
        }
        self.slot_filling = {
            'document_name': None,
            'chapter': None,
            'topic': None,
            'time_range': None
        }

class ConsistencyChecker:
    """答案一致性检查器"""
    
    def __init__(self, model=None):
        self.model = model
        self.consistency_history = []
    
    def check_consistency(self, 
                         current_answer: str, 
                         history_answers: List[str], 
                         question: str) -> Dict[str, Any]:
        """检查答案一致性"""
        logger.info("开始检查答案一致性")
        
        if not history_answers:
            return {
                'is_consistent': True,
                'confidence': 1.0,
                'issues': [],
                'suggestions': []
            }
        
        # 简单的文本相似度检查
        consistency_score = self._calculate_similarity(current_answer, history_answers)
        
        # 检查逻辑一致性
        logical_issues = self._check_logical_consistency(current_answer, history_answers)
        
        # 检查事实一致性
        factual_issues = self._check_factual_consistency(current_answer, history_answers)
        
        is_consistent = consistency_score > 0.7 and not logical_issues and not factual_issues
        
        result = {
            'is_consistent': is_consistent,
            'confidence': consistency_score,
            'issues': logical_issues + factual_issues,
            'suggestions': self._generate_suggestions(consistency_score, logical_issues, factual_issues)
        }
        
        # 记录一致性检查历史
        self.consistency_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'current_answer': current_answer,
            'consistency_result': result
        })
        
        return result
    
    def _calculate_similarity(self, current_answer: str, history_answers: List[str]) -> float:
        """计算答案相似度"""
        if not history_answers:
            return 1.0
        
        # 简单的词汇重叠计算
        current_words = set(current_answer.lower().split())
        total_overlap = 0
        
        for hist_answer in history_answers:
            hist_words = set(hist_answer.lower().split())
            overlap = len(current_words.intersection(hist_words))
            total_overlap += overlap / max(len(current_words), len(hist_words))
        
        return total_overlap / len(history_answers)
    
    def _check_logical_consistency(self, current_answer: str, history_answers: List[str]) -> List[str]:
        """检查逻辑一致性"""
        issues = []
        
        # 检查矛盾词汇
        contradiction_pairs = [
            ('是', '不是'), ('有', '没有'), ('可以', '不可以'),
            ('正确', '错误'), ('存在', '不存在'), ('支持', '反对')
        ]
        
        for pos, neg in contradiction_pairs:
            current_has_pos = pos in current_answer
            current_has_neg = neg in current_answer
            
            for hist_answer in history_answers:
                hist_has_pos = pos in hist_answer
                hist_has_neg = neg in hist_answer
                
                if (current_has_pos and hist_has_neg) or (current_has_neg and hist_has_pos):
                    issues.append(f"逻辑矛盾: {pos} vs {neg}")
        
        return issues
    
    def _check_factual_consistency(self, current_answer: str, history_answers: List[str]) -> List[str]:
        """检查事实一致性"""
        issues = []
        
        # 检查数字一致性
        import re
        current_numbers = re.findall(r'\d+', current_answer)
        
        for hist_answer in history_answers:
            hist_numbers = re.findall(r'\d+', hist_answer)
            
            # 检查是否有相同的数字但含义不同
            common_numbers = set(current_numbers).intersection(set(hist_numbers))
            if len(common_numbers) > 0:
                # 这里可以添加更复杂的数字含义检查
                pass
        
        return issues
    
    def _generate_suggestions(self, 
                            consistency_score: float, 
                            logical_issues: List[str], 
                            factual_issues: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if consistency_score < 0.7:
            suggestions.append("建议检查答案是否与历史回答保持一致")
        
        if logical_issues:
            suggestions.append("发现逻辑矛盾，建议重新审查答案")
        
        if factual_issues:
            suggestions.append("发现事实不一致，建议核实信息来源")
        
        if not suggestions:
            suggestions.append("答案一致性良好")
        
        return suggestions

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, embeddings: Embeddings = None):
        self.memory = ConversationMemory()
        self.state_manager = DialogueStateManager()
        self.consistency_checker = ConsistencyChecker()
        self.embeddings = embeddings
        
        # 对话配置
        self.config = {
            'max_context_length': 2000,
            'enable_memory': True,
            'enable_state_tracking': True,
            'enable_consistency_check': True
        }
    
    def process_query(self, 
                     user_query: str, 
                     system_response: str, 
                     context_docs: List[Document] = None) -> Dict[str, Any]:
        """处理用户查询"""
        logger.info(f"处理用户查询: {user_query}")
        
        # 1. 更新对话状态
        if self.config['enable_state_tracking']:
            self.state_manager.update_state(user_query, context_docs)
        
        # 2. 添加到记忆
        if self.config['enable_memory']:
            self.memory.add_interaction(user_query, system_response, context_docs)
        
        # 3. 检查一致性
        consistency_result = None
        if self.config['enable_consistency_check']:
            history_answers = [interaction['system_response'] 
                             for interaction in self.memory.get_recent_context()]
            consistency_result = self.consistency_checker.check_consistency(
                system_response, history_answers, user_query
            )
        
        # 4. 准备响应
        response = {
            'user_query': user_query,
            'system_response': system_response,
            'context_documents': len(context_docs) if context_docs else 0,
            'conversation_state': self.state_manager.get_current_state(),
            'consistency_check': consistency_result,
            'memory_summary': self._get_memory_summary()
        }
        
        return response
    
    def _get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        recent_context = self.memory.get_recent_context()
        
        return {
            'recent_interactions': len(recent_context),
            'conversation_history_count': len(self.memory.get_conversation_history()),
            'active_topics': list(self.state_manager.current_state['focus_topics']),
            'conversation_mode': self.state_manager.current_state['conversation_mode']
        }
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """为查询获取上下文"""
        context = {
            'recent_conversation': self.memory.get_recent_context(3),
            'current_state': self.state_manager.get_current_state(),
            'conversation_summary': self.memory.conversation_summaries[-1] if self.memory.conversation_summaries else None
        }
        
        return context
    
    def reset_conversation(self):
        """重置对话"""
        self.memory = ConversationMemory()
        self.state_manager.reset_state()
        self.consistency_checker.consistency_history = []
        logger.info("对话已重置")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        return {
            'total_interactions': len(self.memory.get_conversation_history()),
            'current_state': self.state_manager.get_current_state(),
            'consistency_checks': len(self.consistency_checker.consistency_history),
            'memory_usage': {
                'short_term': len(self.memory.short_term_memory),
                'long_term': len(self.memory.conversation_summaries)
            }
        }

