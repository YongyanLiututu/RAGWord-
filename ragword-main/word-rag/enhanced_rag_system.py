import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv

# 导入自定义模块
from enhanced_word_loader import EnhancedWordLoader
from hybrid_retriever import HybridRetriever, EnhancedHybridRetriever, KGEnhancedHybridRetriever
from layered_generator import LayeredGenerator, EnhancedLayeredGenerator
from enhanced_agents import (
    UploadAgent, QueryRewriteAgent, RetrieverAgent, 
    RerankerAgent, ReasonerAgent, AnswerAgent, KGAwareAgent, TaskCoordinatorAgent
)
from conversation_manager import ConversationManager

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EnhancedRAGSystem:
    """增强版RAG系统，整合所有优化功能"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 emb_model: str = "text-embedding-ada-002",
                 use_mistral: bool = False,
                 enable_enhanced_features: bool = True,
                 enable_kg: bool = True,
                 kg_mode: str = "fusion"):
        
        self.model_name = model_name
        self.emb_model = emb_model
        self.use_mistral = use_mistral
        self.enable_enhanced_features = enable_enhanced_features
        self.enable_kg = enable_kg
        self.kg_mode = kg_mode
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化组件
        self._initialize_components()
        
        # 初始化Agent系统
        self._initialize_agents()
        
        # 初始化对话管理器
        self._initialize_conversation_manager()
        
        logger.info("增强版RAG系统初始化完成")
    
    def _initialize_models(self):
        """初始化模型"""
        # 初始化嵌入模型
        if self.use_mistral:
            self.embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=os.getenv("MISTRAL_API_KEY", "api")
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.emb_model,
                openai_api_key=os.getenv("OPENAI_API_KEY", "api")
            )
        
        # 初始化聊天模型
        if self.use_mistral:
            self.chat_model = ChatMistralAI(
                model="mistral-small-latest",
                temperature=0.1,
                mistral_api_key=os.getenv("MISTRAL_API_KEY", "api")
            )
        else:
            self.chat_model = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY", "api")
            )
    
    def _initialize_components(self):
        """初始化核心组件"""
        # 增强版文档加载器
        self.word_loader = EnhancedWordLoader(chunk_size=512, chunk_overlap=50)
        
        # 初始化KG组件
        self.kg_loader = None
        self.kg_retriever = None
        if self.enable_kg:
            try:
                from kg_loader import KnowledgeGraphLoader
                from kg_retriever import KnowledgeGraphRetriever
                self.kg_loader = KnowledgeGraphLoader(
                    model_name=self.model_name,
                    use_mistral=self.use_mistral
                )
                logger.info("KG加载器初始化成功")
            except Exception as e:
                logger.warning(f"KG加载器初始化失败: {e}")
        
        # 混合检索器
        if self.enable_enhanced_features:
            if self.enable_kg and self.kg_loader:
                # 使用KG增强检索器
                self.retriever = KGEnhancedHybridRetriever(
                    embeddings=self.embeddings,
                    llm_model=self.chat_model,
                    kg_retriever=self.kg_retriever,
                    enable_kg=self.enable_kg,
                    kg_weight=0.3
                )
                logger.info("KG增强检索器初始化成功")
            else:
                # 使用普通增强检索器
                self.retriever = EnhancedHybridRetriever(
                    embeddings=self.embeddings,
                    llm_model=self.chat_model
                )
        else:
            self.retriever = HybridRetriever(embeddings=self.embeddings)
        
        # 分层生成器
        if self.enable_enhanced_features:
            self.generator = EnhancedLayeredGenerator(
                model_name=self.model_name,
                max_iterations=3
            )
        else:
            self.generator = LayeredGenerator(model_name=self.model_name)
        
        # 向量存储
        self.vector_store = None
        
        logger.info("核心组件初始化完成")
    
    def _initialize_agents(self):
        """初始化Agent系统"""
        # 创建各个Agent
        self.upload_agent = UploadAgent(self.word_loader)
        self.query_rewrite_agent = QueryRewriteAgent(self.model_name)
        self.retriever_agent = RetrieverAgent(self.retriever)
        self.reranker_agent = RerankerAgent(self.model_name)
        self.reasoner_agent = ReasonerAgent(self.model_name)
        self.answer_agent = AnswerAgent(self.generator)
        
        # KG感知Agent
        if self.enable_kg and self.kg_loader:
            self.kg_agent = KGAwareAgent(
                kg_loader=self.kg_loader,
                kg_retriever=self.kg_retriever,
                model_name=self.model_name,
                use_mistral=self.use_mistral
            )
            logger.info("KG感知Agent初始化成功")
        else:
            self.kg_agent = None
        
        # 任务协调器
        self.coordinator = TaskCoordinatorAgent()
        
        # 注册Agent
        self.coordinator.register_agent("upload", self.upload_agent.process_upload)
        self.coordinator.register_agent("query_rewrite", self.query_rewrite_agent.rewrite_query)
        self.coordinator.register_agent("retrieve", self.retriever_agent.retrieve)
        
        # 注册KG Agent（如果启用）
        if self.kg_agent:
            self.coordinator.register_agent("kg_build", self.kg_agent.build_knowledge_graph)
            self.coordinator.register_agent("kg_retrieve", self.kg_agent.kg_enhanced_retrieval)
            self.coordinator.register_agent("kg_search", self.kg_agent.search_entity_in_kg)
        self.coordinator.register_agent("rerank", self.reranker_agent.rerank)
        self.coordinator.register_agent("reason", self.reasoner_agent.reason)
        self.coordinator.register_agent("answer", self.answer_agent.generate_answer)
        
        logger.info("Agent系统初始化完成")
    
    def _initialize_conversation_manager(self):
        """初始化对话管理器"""
        self.conversation_manager = ConversationManager(self.embeddings)
        logger.info("对话管理器初始化完成")
    
    def upload_documents(self, directory_path: str) -> Dict[str, Any]:
        """上传并处理文档"""
        logger.info(f"开始上传文档: {directory_path}")
        
        try:
            # 使用UploadAgent处理文档
            upload_result = self.coordinator.execute_task("upload", directory_path)
            
            if upload_result['status'] == 'success':
                # 构建向量存储
                documents = upload_result['documents']
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # 更新检索器
                self.retriever.vector_store = self.vector_store
                self.retriever.add_documents(documents)
                
                # 构建知识图谱（如果启用）
                kg_result = None
                if self.enable_kg and self.kg_agent:
                    try:
                        logger.info("开始构建知识图谱...")
                        kg_result = self.coordinator.execute_task("kg_build", documents)
                        
                        # 更新KG检索器
                        if kg_result['status'] == 'success' and kg_result['kg_data']['graph']:
                            self.kg_retriever = KnowledgeGraphRetriever(
                                knowledge_graph=kg_result['kg_data']['graph'],
                                model_name=self.model_name,
                                use_mistral=self.use_mistral
                            )
                            # 更新KG增强检索器
                            if hasattr(self.retriever, 'kg_retriever'):
                                self.retriever.kg_retriever = self.kg_retriever
                            logger.info("知识图谱构建完成")
                    except Exception as e:
                        logger.warning(f"知识图谱构建失败: {e}")
                
                logger.info(f"文档上传完成，共处理 {len(documents)} 个文档块")
                return {
                    'status': 'success',
                    'documents_count': len(documents),
                    'statistics': upload_result['statistics'],
                    'kg_result': kg_result
                }
            else:
                logger.error(f"文档上传失败: {upload_result.get('error', '未知错误')}")
                return upload_result
                
        except Exception as e:
            logger.error(f"文档上传处理失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def ask_question(self, question: str, use_enhanced_features: bool = None) -> Dict[str, Any]:
        """问答功能"""
        if use_enhanced_features is None:
            use_enhanced_features = self.enable_enhanced_features
        
        logger.info(f"开始处理问题: {question}")
        
        try:
            # 1. 查询重写（增强功能）
            if use_enhanced_features:
                conversation_context = self.conversation_manager.get_context_for_query(question)
                rewritten_queries = self.coordinator.execute_task(
                    "query_rewrite", 
                    question, 
                    [ctx['user_query'] for ctx in conversation_context.get('recent_conversation', [])]
                )
                logger.info(f"查询重写完成，生成 {len(rewritten_queries)} 个变体")
            else:
                rewritten_queries = [question]
            
            # 2. 文档检索
            if use_enhanced_features and len(rewritten_queries) > 1:
                # 多查询检索
                retrieved_docs = self.retriever_agent.multi_query_retrieve(rewritten_queries)
            else:
                # 单查询检索
                if self.enable_kg and hasattr(self.retriever, 'kg_enhanced_retrieve'):
                    # 使用KG增强检索
                    retrieved_docs = self.retriever.kg_enhanced_retrieve(question)
                else:
                    # 使用传统检索
                    retrieved_docs = self.coordinator.execute_task("retrieve", question)
            
            if not retrieved_docs:
                return {
                    'status': 'error',
                    'error': '未找到相关文档',
                    'question': question
                }
            
            # 3. 重新排序（增强功能）
            if use_enhanced_features:
                reranked_docs = self.coordinator.execute_task("rerank", question, retrieved_docs)
                final_docs = [doc for doc, _ in reranked_docs]
            else:
                final_docs = [doc for doc, _ in retrieved_docs]
            
            # 4. 推理分析（增强功能）
            reasoning_result = None
            if use_enhanced_features:
                reasoning_result = self.coordinator.execute_task("reason", question, final_docs)
            
            # 5. 生成答案
            answer_result = self.coordinator.execute_task(
                "answer", 
                question, 
                final_docs, 
                reasoning_result
            )
            
            # 6. 更新对话管理器
            self.conversation_manager.process_query(
                question, 
                answer_result['answer'], 
                final_docs
            )
            
            # 7. 准备响应
            response = {
                'status': 'success',
                'question': question,
                'answer': answer_result['answer'],
                'source_documents': len(final_docs),
                'reasoning': reasoning_result,
                'conversation_context': self.conversation_manager.get_conversation_stats(),
                'enhanced_features_used': use_enhanced_features
            }
            
            # 添加摘要信息（如果有）
            if 'summaries' in answer_result:
                response['summaries'] = answer_result['summaries']
            
            # 添加验证信息（如果有）
            if 'verification' in answer_result:
                response['verification'] = answer_result['verification']
            
            logger.info("问题处理完成")
            return response
            
        except Exception as e:
            logger.error(f"问题处理失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'question': question
            }
    
    def multi_turn_conversation(self, questions: List[str]) -> List[Dict[str, Any]]:
        """多轮对话"""
        logger.info(f"开始多轮对话，问题数量: {len(questions)}")
        
        responses = []
        for i, question in enumerate(questions):
            logger.info(f"处理第 {i+1} 个问题: {question}")
            
            response = self.ask_question(question)
            responses.append(response)
            
            # 检查是否有错误
            if response['status'] == 'error':
                logger.warning(f"第 {i+1} 个问题处理失败: {response['error']}")
        
        logger.info("多轮对话完成")
        return responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'model_info': {
                'chat_model': self.model_name,
                'embedding_model': self.emb_model,
                'use_mistral': self.use_mistral
            },
            'agent_status': self.coordinator.get_agent_status(),
            'conversation_stats': self.conversation_manager.get_conversation_stats(),
            'vector_store_info': {
                'initialized': self.vector_store is not None,
                'document_count': len(self.retriever.documents) if self.retriever.documents else 0
            },
            'enhanced_features': {
                'enabled': self.enable_enhanced_features,
                'hybrid_retrieval': True,
                'query_rewriting': True,
                'reranking': True,
                'reasoning': True,
                'layered_generation': True,
                'conversation_memory': True
            },
            'knowledge_graph': {
                'enabled': self.enable_kg,
                'mode': self.kg_mode,
                'loader_initialized': self.kg_loader is not None,
                'retriever_initialized': self.kg_retriever is not None,
                'agent_initialized': self.kg_agent is not None
            }
        }
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_manager.reset_conversation()
        logger.info("对话已重置")
    
    def save_vector_store(self, path: str):
        """保存向量存储"""
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"向量存储已保存到: {path}")
    
    def load_vector_store(self, path: str):
        """加载向量存储"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self.retriever.vector_store = self.vector_store
            logger.info(f"向量存储已从 {path} 加载")
        else:
            logger.warning(f"向量存储路径不存在: {path}")
    
    # KG相关方法
    def get_kg_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        if self.kg_agent:
            return self.kg_agent.get_kg_statistics()
        return {'error': 'KG功能未启用'}
    
    def search_entity_in_kg(self, entity_name: str, entity_type: str = None) -> Dict[str, Any]:
        """在知识图谱中搜索实体"""
        if self.kg_agent:
            return self.coordinator.execute_task("kg_search", entity_name, entity_type)
        return {'error': 'KG功能未启用'}
    
    def kg_enhanced_query(self, query: str) -> Dict[str, Any]:
        """KG增强查询"""
        if self.kg_agent:
            return self.coordinator.execute_task("kg_retrieve", query)
        return {'error': 'KG功能未启用'}
    
    def set_kg_config(self, enable_kg: bool = None, kg_weight: float = None):
        """设置KG配置"""
        if enable_kg is not None:
            self.enable_kg = enable_kg
        if kg_weight is not None and hasattr(self.retriever, 'set_kg_config'):
            self.retriever.set_kg_config(kg_weight=kg_weight)
        
        logger.info(f"KG配置更新: enable_kg={self.enable_kg}")
    
    def export_knowledge_graph(self, format: str = 'json') -> str:
        """导出知识图谱"""
        if self.kg_loader:
            return self.kg_loader.export_graph(format)
        return {'error': 'KG功能未启用'}

# 使用示例
def create_enhanced_rag_system():
    """创建增强版RAG系统实例"""
    return EnhancedRAGSystem(
        model_name="gpt-4",
        emb_model="text-embedding-ada-002",
        use_mistral=False,
        enable_enhanced_features=True,
        enable_kg=True,
        kg_mode="fusion"
    )

if __name__ == "__main__":
    # 创建系统实例
    rag_system = create_enhanced_rag_system()
    
    # 上传文档
    result = rag_system.upload_documents("word")
    print("文档上传结果:", result)
    
    # 获取KG统计信息
    kg_stats = rag_system.get_kg_statistics()
    print("KG统计信息:", kg_stats)
    
    # 问答测试
    response = rag_system.ask_question("什么是LLM？")
    print("问答结果:", response)
    
    # KG增强查询测试
    kg_query = rag_system.kg_enhanced_query("LLM相关的技术")
    print("KG增强查询结果:", kg_query)
    
    # 获取系统状态
    status = rag_system.get_system_status()
    print("系统状态:", status)
