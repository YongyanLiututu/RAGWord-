import os
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """混合检索系统：结合向量检索和关键词检索"""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 vector_store: FAISS = None,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k: int = 20,
                 rerank_k: int = 5):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.k = k
        self.rerank_k = rerank_k
        
        # BM25检索器
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        
        # CrossEncoder用于rerank
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(cross_encoder_model)
            logger.info(f"成功加载CrossEncoder模型: {cross_encoder_model}")
        except Exception as e:
            logger.warning(f"无法加载CrossEncoder模型: {e}")
            self.cross_encoder = None
    
    def add_documents(self, documents: List[Document]):
        """添加文档到检索系统"""
        self.documents = documents
        
        # 为BM25准备tokenized文档
        self.tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self.tokenized_docs.append(tokens)
        
        # 初始化BM25
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(documents)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def vector_retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """向量检索"""
        if not self.vector_store:
            logger.warning("向量存储未初始化")
            return []
        
        k = k or self.k
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"向量检索完成，返回 {len(docs_and_scores)} 个结果")
            return docs_and_scores
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def bm25_retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """BM25关键词检索"""
        if not self.bm25:
            logger.warning("BM25检索器未初始化")
            return []
        
        k = k or self.k
        try:
            query_tokens = self._tokenize(query)
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append((self.documents[idx], float(scores[idx])))
            
            logger.info(f"BM25检索完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []
    
    def reciprocal_rank_fusion(self, 
                              vector_results: List[Tuple[Document, float]], 
                              bm25_results: List[Tuple[Document, float]], 
                              k: int = 60) -> List[Tuple[Document, float]]:
        """使用RRF (Reciprocal Rank Fusion) 融合检索结果"""
        
        doc_to_rank = {}
        
        # 处理向量检索结果
        for rank, (doc, score) in enumerate(vector_results):
            doc_id = doc.page_content[:100]
            if doc_id not in doc_to_rank:
                doc_to_rank[doc_id] = {'doc': doc, 'vector_rank': rank + 1, 'bm25_rank': float('inf')}
        
        # 处理BM25检索结果
        for rank, (doc, score) in enumerate(bm25_results):
            doc_id = doc.page_content[:100]
            if doc_id not in doc_to_rank:
                doc_to_rank[doc_id] = {'doc': doc, 'vector_rank': float('inf'), 'bm25_rank': rank + 1}
            else:
                doc_to_rank[doc_id]['bm25_rank'] = rank + 1
        
        # 计算RRF分数
        rrf_scores = []
        for doc_id, info in doc_to_rank.items():
            rrf_score = 1 / (k + info['vector_rank']) + 1 / (k + info['bm25_rank'])
            rrf_scores.append((info['doc'], rrf_score))
        
        # 按RRF分数排序
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"RRF融合完成，返回 {len(rrf_scores)} 个结果")
        return rrf_scores
    
    def cross_encoder_rerank(self, 
                           query: str, 
                           candidates: List[Tuple[Document, float]], 
                           top_k: int = None) -> List[Tuple[Document, float]]:
        """使用CrossEncoder重新排序"""
        if not self.cross_encoder:
            logger.warning("CrossEncoder未初始化，跳过rerank")
            return candidates[:top_k] if top_k else candidates
        
        top_k = top_k or self.rerank_k
        
        try:
            # 准备CrossEncoder输入
            pairs = [(query, doc.page_content) for doc, _ in candidates]
            
            # 获取CrossEncoder分数
            scores = self.cross_encoder.predict(pairs)
            
            # 重新排序
            reranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"CrossEncoder rerank完成，返回top-{top_k}结果")
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"CrossEncoder rerank失败: {e}")
            return candidates[:top_k]
    
    def retrieve(self, query: str, use_rerank: bool = True) -> List[Tuple[Document, float]]:
        """执行混合检索"""
        logger.info(f"开始混合检索，查询: {query}")
        
        # 1. 向量检索
        vector_results = self.vector_retrieve(query, k=self.k)
        
        # 2. BM25检索
        bm25_results = self.bm25_retrieve(query, k=self.k)
        
        # 3. RRF融合
        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        
        # 4. CrossEncoder rerank（可选）
        if use_rerank and self.cross_encoder:
            final_results = self.cross_encoder_rerank(query, fused_results, top_k=self.rerank_k)
        else:
            final_results = fused_results[:self.rerank_k]
        
        logger.info(f"混合检索完成，最终返回 {len(final_results)} 个结果")
        return final_results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容LangChain接口的方法"""
        results = self.retrieve(query)
        return [doc for doc, score in results]

class QueryRewriter:
    """查询重写器，用于扩展和优化查询"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
    
    def rewrite_query(self, original_query: str, context: str = "") -> List[str]:
        """重写查询，生成多个变体"""
        if not self.llm_model:
            # 如果没有LLM模型，使用简单的规则重写
            return self._rule_based_rewrite(original_query)
        
        try:
            # 使用LLM重写查询
            prompt = f"""
            原始查询: {original_query}
            上下文: {context}
            
            请生成3-5个相关的查询变体，用于提高文档检索的召回率。
            每个查询应该从不同角度表达相同或相关的信息需求。
            
            返回格式：
            1. [查询变体1]
            2. [查询变体2]
            3. [查询变体3]
            """
            
            response = self.llm_model.generate(prompt)
            # 解析响应，提取查询变体
            queries = self._parse_llm_response(response)
            return [original_query] + queries
        except Exception as e:
            logger.error(f"LLM查询重写失败: {e}")
            return self._rule_based_rewrite(original_query)
    
    def _rule_based_rewrite(self, query: str) -> List[str]:
        """基于规则的查询重写"""
        queries = [query]
        
        # 添加同义词
        synonyms = {
            '什么是': ['什么是', '定义', '概念', '含义'],
            '如何': ['如何', '怎么', '方法', '步骤'],
            '为什么': ['为什么', '原因', '理由', '动机']
        }
        
        for original, syns in synonyms.items():
            if original in query:
                for syn in syns:
                    new_query = query.replace(original, syn)
                    if new_query != query:
                        queries.append(new_query)
        
        return queries
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """解析LLM响应，提取查询变体"""
        queries = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or 
                        line.startswith('3.') or line.startswith('4.') or 
                        line.startswith('5.')):
                query = line.split('.', 1)[1].strip()
                if query:
                    queries.append(query)
        
        return queries

class EnhancedHybridRetriever(HybridRetriever):
    """增强版混合检索器，包含查询重写功能"""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 vector_store: FAISS = None,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k: int = 20,
                 rerank_k: int = 5,
                 llm_model=None):
        super().__init__(embeddings, vector_store, cross_encoder_model, k, rerank_k)
        self.query_rewriter = QueryRewriter(llm_model)
    
    def enhanced_retrieve(self, query: str, context: str = "") -> List[Tuple[Document, float]]:
        """增强检索：包含查询重写"""
        logger.info(f"开始增强检索，原始查询: {query}")
        
        # 1. 查询重写
        rewritten_queries = self.query_rewriter.rewrite_query(query, context)
        logger.info(f"查询重写完成，生成 {len(rewritten_queries)} 个查询变体")
        
        # 2. 对每个查询变体进行检索
        all_results = []
        for i, q in enumerate(rewritten_queries):
            results = self.retrieve(q, use_rerank=False)  # 先不rerank
            # 为结果添加查询来源信息
            for doc, score in results:
                doc.metadata['query_source'] = f"query_{i}"
                doc.metadata['original_query'] = query
                doc.metadata['rewritten_query'] = q
            all_results.extend(results)
        
        # 3. 去重和重新排序
        unique_results = self._deduplicate_results(all_results)
        
        # 4. 最终rerank
        final_results = self.cross_encoder_rerank(query, unique_results, top_k=self.rerank_k)
        
        logger.info(f"增强检索完成，最终返回 {len(final_results)} 个结果")
        return final_results
    
    def _deduplicate_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """去重结果"""
        seen = set()
        unique_results = []
        
        for doc, score in results:
            doc_id = doc.page_content[:100]  # 使用内容前100字符作为ID
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc, score))
        
        return unique_results

class KGEnhancedHybridRetriever(EnhancedHybridRetriever):
    """知识图谱增强的混合检索器，集成KG检索通道"""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 vector_store: FAISS = None,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 k: int = 20,
                 rerank_k: int = 5,
                 llm_model=None,
                 kg_retriever=None,
                 enable_kg: bool = True,
                 kg_weight: float = 0.3):
        """
        初始化KG增强检索器
        
        Args:
            embeddings: 嵌入模型
            vector_store: 向量存储
            cross_encoder_model: CrossEncoder模型
            k: 检索数量
            rerank_k: 重排序数量
            llm_model: LLM模型
            kg_retriever: 知识图谱检索器
            enable_kg: 是否启用KG检索
            kg_weight: KG结果权重
        """
        super().__init__(embeddings, vector_store, cross_encoder_model, k, rerank_k, llm_model)
        self.kg_retriever = kg_retriever
        self.enable_kg = enable_kg
        self.kg_weight = kg_weight
        
        logger.info(f"KG增强检索器初始化完成，KG启用: {enable_kg}, KG权重: {kg_weight}")
    
    def kg_enhanced_retrieve(self, query: str, context: str = "") -> List[Tuple[Document, float]]:
        """KG增强检索：结合向量检索、BM25检索和KG检索"""
        logger.info(f"开始KG增强检索，查询: {query}")
        
        # 1. 传统检索（向量 + BM25）
        traditional_results = self.enhanced_retrieve(query, context)
        
        # 2. KG检索（如果启用）
        kg_results = []
        if self.enable_kg and self.kg_retriever:
            try:
                kg_results = self._kg_retrieve(query)
                logger.info(f"KG检索完成，返回 {len(kg_results)} 个结果")
            except Exception as e:
                logger.error(f"KG检索失败: {e}")
        
        # 3. 融合结果
        if kg_results:
            final_results = self._fusion_with_kg(traditional_results, kg_results)
        else:
            final_results = traditional_results
        
        logger.info(f"KG增强检索完成，最终返回 {len(final_results)} 个结果")
        return final_results
    
    def _kg_retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """执行KG检索"""
        if not self.kg_retriever:
            return []
        
        try:
            # 使用KG检索器进行语义查询
            kg_query_result = self.kg_retriever.semantic_query(query, max_results=10)
            
            kg_results = []
            
            # 处理实体结果
            for entity in kg_query_result.get('entities', []):
                # 将KG实体转换为Document格式
                doc_content = f"实体: {entity['name']}\n类型: {entity['type']}\n上下文: {entity['data'].get('context', '')}"
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        'source': 'knowledge_graph',
                        'entity_id': entity['node_id'],
                        'entity_name': entity['name'],
                        'entity_type': entity['type'],
                        'relevance_score': entity['relevance_score'],
                        'kg_context': entity['data'].get('context', '')
                    }
                )
                kg_results.append((doc, entity['relevance_score']))
            
            # 处理关系结果
            for relation in kg_query_result.get('relations', []):
                # 将KG关系转换为Document格式
                doc_content = f"关系: {relation['relation']}\n源实体: {relation['source']}\n目标实体: {relation['target']}\n上下文: {relation['data'].get('context', '')}"
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        'source': 'knowledge_graph',
                        'relation_type': relation['relation'],
                        'source_entity': relation['source'],
                        'target_entity': relation['target'],
                        'kg_context': relation['data'].get('context', '')
                    }
                )
                kg_results.append((doc, 0.8))  # 关系默认分数
            
            return kg_results
            
        except Exception as e:
            logger.error(f"KG检索处理失败: {e}")
            return []
    
    def _fusion_with_kg(self, 
                       traditional_results: List[Tuple[Document, float]], 
                       kg_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """融合传统检索结果和KG检索结果"""
        
        # 创建文档到分数的映射
        doc_scores = {}
        
        # 处理传统检索结果
        for doc, score in traditional_results:
            doc_id = doc.page_content[:100]
            doc_scores[doc_id] = {
                'doc': doc,
                'traditional_score': score,
                'kg_score': 0.0,
                'final_score': score
            }
        
        # 处理KG检索结果
        for doc, kg_score in kg_results:
            doc_id = doc.page_content[:100]
            if doc_id in doc_scores:
                # 如果文档已存在，增加KG分数
                doc_scores[doc_id]['kg_score'] = kg_score
                doc_scores[doc_id]['final_score'] = (
                    doc_scores[doc_id]['traditional_score'] * (1 - self.kg_weight) +
                    kg_score * self.kg_weight
                )
            else:
                # 如果文档不存在，添加新文档
                doc_scores[doc_id] = {
                    'doc': doc,
                    'traditional_score': 0.0,
                    'kg_score': kg_score,
                    'final_score': kg_score * self.kg_weight
                }
        
        # 按最终分数排序
        final_results = []
        for doc_id, info in doc_scores.items():
            final_results.append((info['doc'], info['final_score']))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"KG融合完成，传统结果: {len(traditional_results)}, KG结果: {len(kg_results)}, 融合后: {len(final_results)}")
        
        return final_results[:self.rerank_k]
    
    def get_kg_statistics(self) -> Dict[str, Any]:
        """获取KG统计信息"""
        if self.kg_retriever:
            return self.kg_retriever.get_statistics()
        return {'error': 'KG检索器未初始化'}
    
    def set_kg_config(self, enable_kg: bool = None, kg_weight: float = None):
        """设置KG配置"""
        if enable_kg is not None:
            self.enable_kg = enable_kg
        if kg_weight is not None:
            self.kg_weight = kg_weight
        
        logger.info(f"KG配置更新: enable_kg={self.enable_kg}, kg_weight={self.kg_weight}")
