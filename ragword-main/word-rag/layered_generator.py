import os
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayeredGenerator:
    """分层生成系统：段落级摘要 + 跨段落融合"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化模型
        if "gpt" in model_name.lower():
            self.model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # 使用MistralAI作为备选
            self.model = ChatMistralAI(
                model="mistral-small-latest",
                temperature=temperature
            )
        
        # 定义提示词模板
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的文档摘要助手。请根据给定的文档片段，生成简洁准确的摘要。

要求：
1. 保持原文的核心信息和关键观点
2. 使用简洁明了的语言
3. 摘要长度控制在原文的30%以内
4. 如果原文包含具体数据、日期、人名等，请保留
5. 保持逻辑结构清晰

文档片段：{content}

请生成摘要："""),
            ("human", "{content}")
        ])
        
        self.fusion_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案生成助手。请基于多个文档摘要，生成一个完整、准确的答案。

要求：
1. 综合所有相关信息，避免重复
2. 保持逻辑清晰，结构合理
3. 如果信息有冲突，请说明并给出最可能的解释
4. 答案要直接回答用户问题
5. 在答案末尾列出信息来源

用户问题：{question}
文档摘要：{summaries}

请生成答案："""),
            ("human", "问题：{question}\n\n摘要：{summaries}")
        ])
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个答案验证助手。请检查生成的答案是否准确、完整。

验证标准：
1. 答案是否直接回答了用户问题
2. 答案是否基于提供的文档内容
3. 答案是否有逻辑错误或矛盾
4. 答案是否完整，是否遗漏重要信息

用户问题：{question}
生成的答案：{answer}
原始文档：{documents}

请给出验证结果和改进建议："""),
            ("human", "问题：{question}\n答案：{answer}\n文档：{documents}")
        ])
    
    def generate_paragraph_summaries(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """为每个文档片段生成摘要"""
        logger.info(f"开始生成段落摘要，文档数量: {len(documents)}")
        
        summaries = []
        for i, doc in enumerate(documents):
            try:
                # 生成摘要
                response = self.model.invoke(
                    self.summary_prompt.format(content=doc.page_content)
                )
                
                summary = {
                    'index': i,
                    'original_content': doc.page_content,
                    'summary': response.content,
                    'metadata': doc.metadata,
                    'length_ratio': len(response.content) / len(doc.page_content)
                }
                
                summaries.append(summary)
                logger.info(f"文档 {i+1}/{len(documents)} 摘要生成完成")
                
            except Exception as e:
                logger.error(f"生成文档 {i} 摘要失败: {e}")
                # 如果摘要生成失败，使用原文
                summaries.append({
                    'index': i,
                    'original_content': doc.page_content,
                    'summary': doc.page_content,
                    'metadata': doc.metadata,
                    'length_ratio': 1.0
                })
        
        logger.info(f"段落摘要生成完成，共 {len(summaries)} 个摘要")
        return summaries
    
    def fuse_summaries(self, question: str, summaries: List[Dict[str, Any]]) -> str:
        """融合多个摘要生成最终答案"""
        logger.info(f"开始融合摘要，摘要数量: {len(summaries)}")
        
        try:
            # 准备摘要文本
            summary_texts = []
            for i, summary in enumerate(summaries):
                summary_texts.append(f"摘要{i+1}: {summary['summary']}")
            
            summaries_text = "\n\n".join(summary_texts)
            
            # 生成融合答案
            response = self.model.invoke(
                self.fusion_prompt.format(
                    question=question,
                    summaries=summaries_text
                )
            )
            
            logger.info("摘要融合完成")
            return response.content
            
        except Exception as e:
            logger.error(f"摘要融合失败: {e}")
            # 如果融合失败，返回第一个摘要
            return summaries[0]['summary'] if summaries else "无法生成答案"
    
    def verify_answer(self, question: str, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """验证答案的准确性"""
        logger.info("开始验证答案")
        
        try:
            # 准备文档文本
            doc_texts = []
            for i, doc in enumerate(documents):
                doc_texts.append(f"文档{i+1}: {doc.page_content[:200]}...")
            
            documents_text = "\n\n".join(doc_texts)
            
            # 验证答案
            response = self.model.invoke(
                self.verification_prompt.format(
                    question=question,
                    answer=answer,
                    documents=documents_text
                )
            )
            
            # 解析验证结果
            verification_result = {
                'is_accurate': '准确' in response.content or '正确' in response.content,
                'is_complete': '完整' in response.content,
                'suggestions': response.content,
                'confidence': self._extract_confidence(response.content)
            }
            
            logger.info("答案验证完成")
            return verification_result
            
        except Exception as e:
            logger.error(f"答案验证失败: {e}")
            return {
                'is_accurate': True,
                'is_complete': True,
                'suggestions': "验证过程出现错误",
                'confidence': 0.7
            }
    
    def _extract_confidence(self, verification_text: str) -> float:
        """从验证文本中提取置信度"""
        # 简单的置信度提取逻辑
        if '非常准确' in verification_text or '完全正确' in verification_text:
            return 0.9
        elif '准确' in verification_text or '正确' in verification_text:
            return 0.8
        elif '基本准确' in verification_text:
            return 0.7
        elif '部分准确' in verification_text:
            return 0.6
        else:
            return 0.5
    
    def generate_layered_answer(self, 
                              question: str, 
                              documents: List[Document],
                              verify: bool = True) -> Dict[str, Any]:
        """分层生成答案的完整流程"""
        logger.info(f"开始分层生成答案，问题: {question}")
        
        # 1. 生成段落摘要
        summaries = self.generate_paragraph_summaries(documents)
        
        # 2. 融合摘要生成答案
        answer = self.fuse_summaries(question, summaries)
        
        # 3. 验证答案（可选）
        verification_result = None
        if verify:
            verification_result = self.verify_answer(question, answer, documents)
        
        # 4. 准备结果
        result = {
            'question': question,
            'answer': answer,
            'summaries': summaries,
            'verification': verification_result,
            'source_documents': len(documents),
            'generation_method': 'layered'
        }
        
        logger.info("分层生成答案完成")
        return result

class EnhancedLayeredGenerator(LayeredGenerator):
    """增强版分层生成器，支持多轮优化"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 max_iterations: int = 3):
        super().__init__(model_name, temperature, max_tokens)
        self.max_iterations = max_iterations
        
        # 增强的融合提示词
        self.enhanced_fusion_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案生成助手。请基于多个文档摘要，生成一个完整、准确的答案。

要求：
1. 综合所有相关信息，避免重复和矛盾
2. 保持逻辑清晰，结构合理
3. 如果信息有冲突，请说明并给出最可能的解释
4. 答案要直接回答用户问题
5. 在答案末尾列出信息来源
6. 如果答案不够完整，请指出还需要哪些信息

用户问题：{question}
文档摘要：{summaries}
历史答案：{history}

请生成改进的答案："""),
            ("human", "问题：{question}\n\n摘要：{summaries}\n\n历史答案：{history}")
        ])
    
    def iterative_generation(self, 
                           question: str, 
                           documents: List[Document],
                           initial_answer: str = None) -> Dict[str, Any]:
        """迭代式答案生成"""
        logger.info(f"开始迭代式答案生成，最大迭代次数: {self.max_iterations}")
        
        # 1. 生成初始摘要
        summaries = self.generate_paragraph_summaries(documents)
        
        # 2. 生成初始答案
        if initial_answer:
            current_answer = initial_answer
        else:
            current_answer = self.fuse_summaries(question, summaries)
        
        # 3. 迭代优化
        history = []
        for iteration in range(self.max_iterations):
            logger.info(f"开始第 {iteration + 1} 轮迭代")
            
            # 验证当前答案
            verification = self.verify_answer(question, current_answer, documents)
            
            # 如果答案已经很好，停止迭代
            if verification['confidence'] > 0.8:
                logger.info(f"答案质量已达标，置信度: {verification['confidence']}")
                break
            
            # 记录历史
            history.append({
                'iteration': iteration + 1,
                'answer': current_answer,
                'verification': verification
            })
            
            # 生成改进的答案
            try:
                history_text = "\n".join([f"第{h['iteration']}轮: {h['answer']}" for h in history])
                
                response = self.model.invoke(
                    self.enhanced_fusion_prompt.format(
                        question=question,
                        summaries="\n\n".join([f"摘要{i+1}: {s['summary']}" for i, s in enumerate(summaries)]),
                        history=history_text
                    )
                )
                
                current_answer = response.content
                logger.info(f"第 {iteration + 1} 轮迭代完成")
                
            except Exception as e:
                logger.error(f"第 {iteration + 1} 轮迭代失败: {e}")
                break
        
        # 4. 最终验证
        final_verification = self.verify_answer(question, current_answer, documents)
        
        result = {
            'question': question,
            'final_answer': current_answer,
            'summaries': summaries,
            'verification': final_verification,
            'iteration_history': history,
            'total_iterations': len(history) + 1,
            'source_documents': len(documents),
            'generation_method': 'iterative_layered'
        }
        
        logger.info(f"迭代式答案生成完成，共 {result['total_iterations']} 轮")
        return result

class KGEnhancedLayeredGenerator(EnhancedLayeredGenerator):
    """知识图谱增强的分层生成器，包含KG一致性检查"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 max_iterations: int = 3,
                 kg_loader=None,
                 enable_kg_verification: bool = True):
        """
        初始化KG增强生成器
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            max_iterations: 最大迭代次数
            kg_loader: 知识图谱加载器
            enable_kg_verification: 是否启用KG验证
        """
        super().__init__(model_name, temperature, max_tokens, max_iterations)
        self.kg_loader = kg_loader
        self.enable_kg_verification = enable_kg_verification
        
        # KG一致性检查提示词
        self.kg_consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个知识图谱一致性检查专家。请检查生成的答案与知识图谱中的实体和关系是否一致。

检查标准：
1. 答案中提到的实体是否在知识图谱中存在
2. 实体间的关系是否与知识图谱中的关系一致
3. 答案中的事实是否与知识图谱中的信息相符
4. 是否存在知识图谱中未提及但答案中声称存在的关系

请以JSON格式返回结果：
{
    "consistency_score": 一致性分数(0-1),
    "entity_verification": {
        "verified_entities": ["已验证的实体"],
        "unverified_entities": ["未验证的实体"],
        "missing_entities": ["知识图谱中缺失的实体"]
    },
    "relation_verification": {
        "verified_relations": ["已验证的关系"],
        "unverified_relations": ["未验证的关系"],
        "contradictory_relations": ["矛盾的关系"]
    },
    "overall_assessment": "整体评估",
    "suggestions": ["改进建议"]
}"""),
            ("human", "答案：{answer}\n知识图谱信息：{kg_info}")
        ])
        
        logger.info("KG增强生成器初始化完成")
    
    def generate_layered_answer_with_kg_verification(self, 
                                                   question: str, 
                                                   documents: List[Document],
                                                   kg_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成分层答案并进行KG一致性检查
        
        Args:
            question: 用户问题
            documents: 文档列表
            kg_info: 知识图谱信息
            
        Returns:
            包含KG验证结果的答案
        """
        logger.info(f"开始KG增强的答案生成: {question}")
        
        # 1. 生成基础答案
        base_result = self.iterative_generation(question, documents)
        
        # 2. KG一致性检查（如果启用）
        kg_verification = None
        if self.enable_kg_verification and kg_info:
            try:
                kg_verification = self._check_kg_consistency(base_result['final_answer'], kg_info)
                logger.info(f"KG一致性检查完成，分数: {kg_verification.get('consistency_score', 0)}")
            except Exception as e:
                logger.warning(f"KG一致性检查失败: {e}")
                kg_verification = {
                    'consistency_score': 0.0,
                    'error': str(e)
                }
        
        # 3. 如果KG验证发现问题，尝试改进答案
        improved_answer = base_result['final_answer']
        if kg_verification and kg_verification.get('consistency_score', 1.0) < 0.7:
            try:
                improved_answer = self._improve_answer_with_kg(
                    question, 
                    base_result['final_answer'], 
                    kg_info, 
                    kg_verification
                )
                logger.info("基于KG验证结果改进了答案")
            except Exception as e:
                logger.warning(f"答案改进失败: {e}")
        
        # 4. 构建最终结果
        result = {
            'question': question,
            'answer': improved_answer,
            'base_answer': base_result['final_answer'],
            'summaries': base_result['summaries'],
            'verification': base_result['verification'],
            'kg_verification': kg_verification,
            'iteration_history': base_result['iteration_history'],
            'total_iterations': base_result['total_iterations'],
            'source_documents': len(documents),
            'generation_method': 'kg_enhanced_layered',
            'kg_enhancement_applied': kg_verification is not None
        }
        
        logger.info("KG增强答案生成完成")
        return result
    
    def _check_kg_consistency(self, answer: str, kg_info: Dict[str, Any]) -> Dict[str, Any]:
        """检查答案与知识图谱的一致性"""
        if not self.kg_loader:
            return {'consistency_score': 0.0, 'error': 'KG加载器未初始化'}
        
        try:
            # 从答案中提取实体
            extracted_entities = self._extract_entities_from_answer(answer)
            
            # 构建KG信息文本
            kg_text = self._build_kg_info_text(kg_info)
            
            # 使用LLM进行一致性检查
            response = self.model.invoke(
                self.kg_consistency_prompt.format(
                    answer=answer,
                    kg_info=kg_text
                )
            )
            
            # 解析响应
            import json
            verification_result = json.loads(response.content)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"KG一致性检查失败: {e}")
            return {
                'consistency_score': 0.0,
                'error': str(e)
            }
    
    def _extract_entities_from_answer(self, answer: str) -> List[str]:
        """从答案中提取实体"""
        # 简单的实体提取（可以进一步优化）
        entities = []
        
        # 提取可能的实体（大写开头的词）
        import re
        potential_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', answer)
        entities.extend(potential_entities)
        
        # 提取引号中的内容
        quoted_entities = re.findall(r'"([^"]*)"', answer)
        entities.extend(quoted_entities)
        
        return list(set(entities))
    
    def _build_kg_info_text(self, kg_info: Dict[str, Any]) -> str:
        """构建知识图谱信息文本"""
        kg_text = "知识图谱信息：\n"
        
        if 'entities' in kg_info:
            kg_text += "实体：\n"
            for entity in kg_info['entities'][:10]:  # 限制数量
                kg_text += f"- {entity.get('name', '')} ({entity.get('type', '')})\n"
        
        if 'relations' in kg_info:
            kg_text += "关系：\n"
            for relation in kg_info['relations'][:10]:  # 限制数量
                kg_text += f"- {relation.get('source', '')} --{relation.get('relation', '')}--> {relation.get('target', '')}\n"
        
        if 'statistics' in kg_info:
            stats = kg_info['statistics']
            kg_text += f"统计信息：节点数 {stats.get('node_count', 0)}, 边数 {stats.get('edge_count', 0)}\n"
        
        return kg_text
    
    def _improve_answer_with_kg(self, 
                               question: str, 
                               original_answer: str, 
                               kg_info: Dict[str, Any],
                               verification: Dict[str, Any]) -> str:
        """基于KG验证结果改进答案"""
        
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个答案改进专家。请基于知识图谱验证结果，改进原始答案。

改进要求：
1. 修正与知识图谱不一致的信息
2. 添加知识图谱中相关但答案中缺失的信息
3. 保持答案的逻辑性和可读性
4. 明确标注哪些信息来自知识图谱验证

原始答案：{original_answer}
KG验证结果：{verification}
知识图谱信息：{kg_info}

请生成改进后的答案："""),
            ("human", "问题：{question}\n原始答案：{original_answer}\n验证结果：{verification}\nKG信息：{kg_info}")
        ])
        
        try:
            response = self.model.invoke(
                improvement_prompt.format(
                    question=question,
                    original_answer=original_answer,
                    verification=str(verification),
                    kg_info=self._build_kg_info_text(kg_info)
                )
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"答案改进失败: {e}")
            return original_answer
    
    def get_kg_verification_stats(self) -> Dict[str, Any]:
        """获取KG验证统计信息"""
        return {
            'kg_verification_enabled': self.enable_kg_verification,
            'kg_loader_available': self.kg_loader is not None
        }
