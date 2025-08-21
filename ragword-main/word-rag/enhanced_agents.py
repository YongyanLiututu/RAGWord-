import os
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from enhanced_word_loader import EnhancedWordLoader
from hybrid_retriever import HybridRetriever
from layered_generator import LayeredGenerator
from kg_loader import KnowledgeGraphLoader
from kg_retriever import KnowledgeGraphRetriever
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadAgent:
    """文档上传处理Agent"""
    
    def __init__(self, enhanced_loader: EnhancedWordLoader):
        self.loader = enhanced_loader
        self.upload_history = []
    
    def process_upload(self, file_path: str) -> Dict[str, Any]:
        """处理文档上传"""
        logger.info(f"UploadAgent开始处理文档: {file_path}")
        
        try:
            # 加载和切分文档
            result = self.loader.load(file_path)
            
            # 记录上传历史
            upload_record = {
                'file_path': file_path,
                'upload_time': datetime.now().isoformat(),
                'doc_id': result['metadata']['doc_id'],
                'total_chunks': result['statistics']['total_chunks'],
                'status': 'success'
            }
            self.upload_history.append(upload_record)
            
            logger.info(f"文档上传处理完成: {file_path}")
            return {
                'status': 'success',
                'documents': result['documents'],
                'structure': result['structure'],
                'metadata': result['metadata'],
                'statistics': result['statistics']
            }
            
        except Exception as e:
            logger.error(f"文档上传处理失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_upload_history(self) -> List[Dict[str, Any]]:
        """获取上传历史"""
        return self.upload_history

class QueryRewriteAgent:
    """查询重写Agent"""
    
    def __init__(self, model_name: str = "gpt-4"):
        if "gpt" in model_name.lower():
            self.model = ChatOpenAI(model=model_name, temperature=0.1)
        else:
            self.model = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
        
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。请根据用户的问题，生成多个相关的查询变体，以提高文档检索的召回率。

要求：
1. 保持原始问题的核心意图
2. 生成3-5个不同的查询变体
3. 考虑同义词、相关概念、不同表达方式
4. 每个变体应该从不同角度表达相同或相关的信息需求

原始查询：{original_query}
对话历史：{conversation_history}

请生成查询变体："""),
            ("human", "原始查询：{original_query}\n对话历史：{conversation_history}")
        ])
    
    def rewrite_query(self, original_query: str, conversation_history: List[str] = None) -> List[str]:
        """重写查询"""
        logger.info(f"QueryRewriteAgent开始重写查询: {original_query}")
        
        try:
            # 准备对话历史
            history_text = ""
            if conversation_history:
                history_text = "\n".join([f"用户: {msg}" for msg in conversation_history[-3:]])  # 只保留最近3轮
            
            # 生成查询变体
            response = self.model.invoke(
                self.rewrite_prompt.format(
                    original_query=original_query,
                    conversation_history=history_text
                )
            )
            
            # 解析响应，提取查询变体
            queries = self._parse_rewritten_queries(response.content)
            queries.insert(0, original_query)  # 添加原始查询
            
            logger.info(f"查询重写完成，生成 {len(queries)} 个查询变体")
            return queries
            
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return [original_query]
    
    def _parse_rewritten_queries(self, response: str) -> List[str]:
        """解析重写的查询"""
        queries = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配数字编号的查询
            if re.match(r'^\d+[\.\)]', line):
                query = re.sub(r'^\d+[\.\)]\s*', '', line)
                if query:
                    queries.append(query)
        
        return queries

class RetrieverAgent:
    """检索Agent"""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
        self.retrieval_history = []
    
    def retrieve(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """执行检索"""
        logger.info(f"RetrieverAgent开始检索: {query}")
        
        try:
            # 执行混合检索
            results = self.retriever.retrieve(query)
            
            # 记录检索历史
            retrieval_record = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'results_count': len(results),
                'top_score': results[0][1] if results else 0
            }
            self.retrieval_history.append(retrieval_record)
            
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def multi_query_retrieve(self, queries: List[str], k: int = 20) -> List[Tuple[Document, float]]:
        """多查询检索"""
        logger.info(f"RetrieverAgent开始多查询检索，查询数量: {len(queries)}")
        
        all_results = []
        for i, query in enumerate(queries):
            results = self.retrieve(query, k)
            # 为结果添加查询来源信息
            for doc, score in results:
                doc.metadata['query_source'] = f"query_{i}"
                doc.metadata['original_query'] = query
            all_results.extend(results)
        
        # 去重和重新排序
        unique_results = self._deduplicate_results(all_results)
        
        logger.info(f"多查询检索完成，去重后返回 {len(unique_results)} 个结果")
        return unique_results
    
    def _deduplicate_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """去重结果"""
        seen = set()
        unique_results = []
        
        for doc, score in results:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc, score))
        
        # 按分数排序
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results

class RerankerAgent:
    """重新排序Agent"""
    
    def __init__(self, model_name: str = "gpt-4"):
        if "gpt" in model_name.lower():
            self.model = ChatOpenAI(model=model_name, temperature=0.1)
        else:
            self.model = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
        
        self.rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个文档相关性评估专家。请根据用户问题，评估每个文档片段的相关性，并给出0-10的分数。

评估标准：
1. 内容相关性：文档是否直接回答或支持回答问题
2. 信息完整性：文档是否包含完整的信息
3. 时效性：信息是否最新和准确
4. 可信度：信息来源是否可靠

用户问题：{question}

请评估以下文档片段："""),
            ("human", "问题：{question}\n\n文档：{documents}")
        ])
    
    def rerank(self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """重新排序候选文档"""
        logger.info(f"RerankerAgent开始重新排序，候选文档数量: {len(candidates)}")
        
        try:
            # 准备文档文本
            doc_texts = []
            for i, (doc, score) in enumerate(candidates):
                doc_texts.append(f"文档{i+1}: {doc.page_content[:300]}...")
            
            documents_text = "\n\n".join(doc_texts)
            
            # 使用LLM评估相关性
            response = self.model.invoke(
                self.rerank_prompt.format(
                    question=query,
                    documents=documents_text
                )
            )
            
            # 解析评估结果
            scores = self._parse_rerank_scores(response.content, len(candidates))
            
            # 重新排序
            reranked = []
            for i, (doc, original_score) in enumerate(candidates):
                new_score = scores[i] if i < len(scores) else original_score
                reranked.append((doc, new_score))
            
            # 按新分数排序
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"重新排序完成，返回top-{top_k}结果")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"重新排序失败: {e}")
            return candidates[:top_k]
    
    def _parse_rerank_scores(self, response: str, expected_count: int) -> List[float]:
        """解析重新排序分数"""
        scores = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配分数模式
            match = re.search(r'文档\d+.*?(\d+(?:\.\d+)?)', line)
            if match:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                except ValueError:
                    continue
        
        # 如果解析的分数不够，用默认分数填充
        while len(scores) < expected_count:
            scores.append(5.0)
        
        return scores[:expected_count]

class ReasonerAgent:
    """推理Agent"""
    
    def __init__(self, model_name: str = "gpt-4"):
        if "gpt" in model_name.lower():
            self.model = ChatOpenAI(model=model_name, temperature=0.1)
        else:
            self.model = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
        
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个逻辑推理专家。请基于提供的文档内容，进行深度推理分析，确保答案的逻辑性和一致性。

推理要求：
1. 分析文档中的因果关系
2. 识别隐含的信息和逻辑
3. 检查信息的一致性和矛盾
4. 提供推理过程和证据
5. 指出不确定或需要进一步验证的信息

用户问题：{question}
文档内容：{documents}

请进行推理分析："""),
            ("human", "问题：{question}\n\n文档：{documents}")
        ])
    
    def reason(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """执行推理分析"""
        logger.info(f"ReasonerAgent开始推理分析: {question}")
        
        try:
            # 准备文档内容
            doc_texts = []
            for i, doc in enumerate(documents):
                doc_texts.append(f"文档{i+1}: {doc.page_content}")
            
            documents_text = "\n\n".join(doc_texts)
            
            # 执行推理
            response = self.model.invoke(
                self.reasoning_prompt.format(
                    question=question,
                    documents=documents_text
                )
            )
            
            # 分析推理结果
            reasoning_result = {
                'question': question,
                'reasoning_process': response.content,
                'confidence': self._extract_confidence(response.content),
                'logical_consistency': self._check_consistency(response.content),
                'evidence_quality': self._assess_evidence_quality(response.content)
            }
            
            logger.info("推理分析完成")
            return reasoning_result
            
        except Exception as e:
            logger.error(f"推理分析失败: {e}")
            return {
                'question': question,
                'reasoning_process': "推理过程出现错误",
                'confidence': 0.5,
                'logical_consistency': False,
                'evidence_quality': 'low'
            }
    
    def _extract_confidence(self, reasoning_text: str) -> float:
        """提取置信度"""
        if '高度确信' in reasoning_text or '非常确定' in reasoning_text:
            return 0.9
        elif '确信' in reasoning_text or '确定' in reasoning_text:
            return 0.8
        elif '基本确定' in reasoning_text:
            return 0.7
        elif '可能' in reasoning_text:
            return 0.6
        else:
            return 0.5
    
    def _check_consistency(self, reasoning_text: str) -> bool:
        """检查逻辑一致性"""
        inconsistent_keywords = ['矛盾', '不一致', '冲突', '相反', '对立']
        return not any(keyword in reasoning_text for keyword in inconsistent_keywords)
    
    def _assess_evidence_quality(self, reasoning_text: str) -> str:
        """评估证据质量"""
        if '充分证据' in reasoning_text or '明确证据' in reasoning_text:
            return 'high'
        elif '部分证据' in reasoning_text or '有限证据' in reasoning_text:
            return 'medium'
        else:
            return 'low'

class AnswerAgent:
    """答案生成Agent"""
    
    def __init__(self, layered_generator: LayeredGenerator):
        self.generator = layered_generator
        self.answer_history = []
    
    def generate_answer(self, 
                       question: str, 
                       documents: List[Document],
                       reasoning_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成最终答案"""
        logger.info(f"AnswerAgent开始生成答案: {question}")
        
        try:
            # 使用分层生成器生成答案
            result = self.generator.generate_layered_answer(question, documents)
            
            # 添加推理信息
            if reasoning_result:
                result['reasoning'] = reasoning_result
            
            # 记录答案历史
            answer_record = {
                'question': question,
                'answer': result['answer'],
                'timestamp': datetime.now().isoformat(),
                'source_documents': len(documents),
                'verification_confidence': result.get('verification', {}).get('confidence', 0)
            }
            self.answer_history.append(answer_record)
            
            logger.info("答案生成完成")
            return result
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                'question': question,
                'answer': "抱歉，生成答案时出现错误",
                'error': str(e)
            }
    
    def get_answer_history(self) -> List[Dict[str, Any]]:
        """获取答案历史"""
        return self.answer_history

class KGAwareAgent:
    """知识图谱感知Agent，专门处理KG检索和验证"""
    
    def __init__(self, 
                 kg_loader: KnowledgeGraphLoader = None,
                 kg_retriever: KnowledgeGraphRetriever = None,
                 model_name: str = "gpt-4",
                 use_mistral: bool = False):
        """
        初始化KG感知Agent
        
        Args:
            kg_loader: 知识图谱加载器
            kg_retriever: 知识图谱检索器
            model_name: 使用的LLM模型名称
            use_mistral: 是否使用MistralAI模型
        """
        self.kg_loader = kg_loader
        self.kg_retriever = kg_retriever
        self.model_name = model_name
        self.use_mistral = use_mistral
        
        # 初始化LLM
        self._initialize_llm()
        
        # KG处理历史
        self.kg_history = []
        
        logger.info("KGAwareAgent初始化完成")
    
    def _initialize_llm(self):
        """初始化LLM模型"""
        try:
            if self.use_mistral:
                self.llm = ChatMistralAI(
                    model="mistral-small-latest",
                    temperature=0.1,
                    mistral_api_key=os.getenv("MISTRAL_API_KEY", "api")
                )
            else:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=0.1,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "api")
                )
            logger.info(f"KGAwareAgent LLM模型初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"KGAwareAgent LLM模型初始化失败: {e}")
            self.llm = None
    
    def build_knowledge_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """
        构建知识图谱
        
        Args:
            documents: 文档列表
            
        Returns:
            知识图谱构建结果
        """
        logger.info(f"KGAwareAgent开始构建知识图谱，文档数量: {len(documents)}")
        
        if not self.kg_loader:
            logger.warning("KG加载器未初始化，无法构建知识图谱")
            return {'status': 'error', 'error': 'KG加载器未初始化'}
        
        try:
            # 使用KG加载器抽取实体和关系
            kg_data = self.kg_loader.extract_entities_and_relations(documents)
            
            # 初始化KG检索器
            if self.kg_retriever is None and kg_data['graph']:
                self.kg_retriever = KnowledgeGraphRetriever(
                    knowledge_graph=kg_data['graph'],
                    model_name=self.model_name,
                    use_mistral=self.use_mistral
                )
            
            # 记录构建历史
            build_record = {
                'timestamp': datetime.now().isoformat(),
                'document_count': len(documents),
                'entity_count': kg_data['statistics']['node_count'],
                'relation_count': kg_data['statistics']['edge_count'],
                'entity_types': kg_data['statistics']['entity_types'],
                'relation_types': kg_data['statistics']['relation_types']
            }
            self.kg_history.append(build_record)
            
            logger.info(f"知识图谱构建完成: {kg_data['statistics']['node_count']} 个实体, {kg_data['statistics']['edge_count']} 个关系")
            
            return {
                'status': 'success',
                'kg_data': kg_data,
                'statistics': kg_data['statistics']
            }
            
        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def kg_enhanced_retrieval(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        KG增强检索
        
        Args:
            query: 查询字符串
            max_results: 最大返回结果数
            
        Returns:
            KG检索结果
        """
        logger.info(f"KGAwareAgent开始KG增强检索: {query}")
        
        if not self.kg_retriever:
            logger.warning("KG检索器未初始化，无法进行KG检索")
            return {'status': 'error', 'error': 'KG检索器未初始化'}
        
        try:
            # 执行KG语义查询
            kg_results = self.kg_retriever.semantic_query(query, max_results)
            
            # 分析查询意图
            intent_analysis = self._analyze_query_intent(query, kg_results)
            
            # 验证KG结果
            verification_result = self._verify_kg_results(query, kg_results)
            
            result = {
                'status': 'success',
                'query': query,
                'kg_results': kg_results,
                'intent_analysis': intent_analysis,
                'verification': verification_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # 记录检索历史
            retrieval_record = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'entity_count': len(kg_results.get('entities', [])),
                'relation_count': len(kg_results.get('relations', [])),
                'path_count': len(kg_results.get('paths', [])),
                'intent_type': intent_analysis.get('type', 'unknown')
            }
            self.kg_history.append(retrieval_record)
            
            logger.info(f"KG增强检索完成: {len(kg_results.get('entities', []))} 个实体, {len(kg_results.get('relations', []))} 个关系")
            
            return result
            
        except Exception as e:
            logger.error(f"KG增强检索失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_query_intent(self, query: str, kg_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析查询意图"""
        if not self.llm:
            return {'type': 'unknown', 'confidence': 0.0}
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个查询意图分析专家。请分析用户查询的意图类型和相关信息。

意图类型包括：
1. entity_search: 搜索特定实体
2. relation_search: 搜索实体间关系
3. path_search: 搜索实体间路径
4. concept_exploration: 概念探索
5. fact_verification: 事实验证

请以JSON格式返回结果：
{
    "type": "意图类型",
    "confidence": 置信度(0-1),
    "target_entities": ["目标实体列表"],
    "relation_focus": "关系焦点",
    "explanation": "意图解释"
}"""),
                ("human", f"查询: {query}\nKG结果: {kg_results}")
            ])
            
            response = self.llm.invoke(prompt.format(query=query, kg_results=kg_results))
            intent = json.loads(response.content)
            return intent
            
        except Exception as e:
            logger.error(f"查询意图分析失败: {e}")
            return {'type': 'unknown', 'confidence': 0.0}
    
    def _verify_kg_results(self, query: str, kg_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证KG结果的相关性和准确性"""
        if not self.llm:
            return {'relevance_score': 0.0, 'accuracy_score': 0.0, 'verification': 'LLM未初始化'}
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个知识图谱结果验证专家。请评估KG检索结果与查询的相关性和准确性。

请以JSON格式返回结果：
{
    "relevance_score": 相关性分数(0-1),
    "accuracy_score": 准确性分数(0-1),
    "verification": "验证说明",
    "suggestions": ["改进建议"]
}"""),
                ("human", f"查询: {query}\nKG结果: {kg_results}")
            ])
            
            response = self.llm.invoke(prompt.format(query=query, kg_results=kg_results))
            verification = json.loads(response.content)
            return verification
            
        except Exception as e:
            logger.error(f"KG结果验证失败: {e}")
            return {'relevance_score': 0.0, 'accuracy_score': 0.0, 'verification': '验证失败'}
    
    def get_kg_statistics(self) -> Dict[str, Any]:
        """获取KG统计信息"""
        if self.kg_retriever:
            return self.kg_retriever.get_statistics()
        return {'error': 'KG检索器未初始化'}
    
    def get_kg_history(self) -> List[Dict[str, Any]]:
        """获取KG处理历史"""
        return self.kg_history
    
    def search_entity_in_kg(self, entity_name: str, entity_type: str = None) -> Dict[str, Any]:
        """在KG中搜索特定实体"""
        if not self.kg_retriever:
            return {'status': 'error', 'error': 'KG检索器未初始化'}
        
        try:
            # 搜索实体
            entities = self.kg_retriever.query_entities(entity_name, entity_type)
            
            # 获取实体的邻居信息
            neighbors_info = []
            for entity in entities:
                if self.kg_loader:
                    neighbors = self.kg_loader.get_entity_neighbors(entity['name'])
                    neighbors_info.append({
                        'entity': entity,
                        'neighbors': neighbors
                    })
            
            return {
                'status': 'success',
                'entities': entities,
                'neighbors_info': neighbors_info
            }
            
        except Exception as e:
            logger.error(f"实体搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}

class TaskCoordinatorAgent:
    """任务协调Agent"""
    
    def __init__(self):
        self.agents = {}
        self.execution_history = []
    
    def register_agent(self, agent_name: str, agent_instance):
        """注册Agent"""
        self.agents[agent_name] = agent_instance
        logger.info(f"注册Agent: {agent_name}")
    
    def execute_task(self, task_name: str, *args, **kwargs) -> Any:
        """执行任务"""
        if task_name not in self.agents:
            raise ValueError(f"未注册的Agent: {task_name}")
        
        logger.info(f"执行任务: {task_name}")
        start_time = datetime.now()
        
        try:
            result = self.agents[task_name](*args, **kwargs)
            
            # 记录执行历史
            execution_record = {
                'task_name': task_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'success',
                'result_summary': str(result)[:100] if result else ''
            }
            self.execution_history.append(execution_record)
            
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {task_name}, 错误: {e}")
            
            # 记录失败历史
            execution_record = {
                'task_name': task_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            self.execution_history.append(execution_record)
            
            raise e
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            'registered_agents': list(self.agents.keys()),
            'total_executions': len(self.execution_history),
            'successful_executions': len([e for e in self.execution_history if e['status'] == 'success']),
            'failed_executions': len([e for e in self.execution_history if e['status'] == 'error'])
        }
