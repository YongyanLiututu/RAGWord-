import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import networkx as nx
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphLoader:
    """知识图谱加载器：从文档中抽取实体和关系"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 use_mistral: bool = False):
        """
        初始化知识图谱加载器
        
        Args:
            model_name: 使用的LLM模型名称
            use_mistral: 是否使用MistralAI模型
        """
        self.model_name = model_name
        self.use_mistral = use_mistral
        
        # 初始化LLM
        self._initialize_llm()
        
        # 知识图谱存储
        self.knowledge_graph = nx.DiGraph()
        self.entity_mapping = {}  # 实体标准化映射
        self.relation_types = set()  # 关系类型集合
        
        # 实体类型定义
        self.entity_types = {
            'PERSON': '人物',
            'ORG': '机构',
            'LOC': '地点',
            'TIME': '时间',
            'CONCEPT': '概念',
            'EVENT': '事件',
            'TECH': '技术',
            'PRODUCT': '产品'
        }
        
        logger.info("知识图谱加载器初始化完成")
    
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
            logger.info(f"LLM模型初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM模型初始化失败: {e}")
            self.llm = None
    
    def extract_entities_and_relations(self, documents: List[Document]) -> Dict[str, Any]:
        """
        从文档中抽取实体和关系
        
        Args:
            documents: 文档列表
            
        Returns:
            包含实体和关系的字典
        """
        logger.info(f"开始从 {len(documents)} 个文档中抽取实体和关系")
        
        all_entities = []
        all_relations = []
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}: {doc.metadata.get('source', 'unknown')}")
            
            # 使用LLM抽取实体和关系
            result = self._extract_with_llm(doc.page_content, doc.metadata)
            
            all_entities.extend(result['entities'])
            all_relations.extend(result['relations'])
        
        # 构建知识图谱
        kg_data = self._build_knowledge_graph(all_entities, all_relations)
        
        logger.info(f"实体抽取完成: {len(all_entities)} 个实体, {len(all_relations)} 个关系")
        
        return kg_data
    
    def _extract_with_llm(self, text: str, metadata: Dict[str, Any]) -> Dict[str, List]:
        """使用LLM抽取实体和关系"""
        if not self.llm:
            return {'entities': [], 'relations': []}
        
        # 限制文本长度以避免token超限
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的知识图谱构建专家。请从给定的文本中抽取实体和关系。

实体类型包括：
- PERSON: 人物（人名、职位等）
- ORG: 机构（公司、组织、部门等）
- LOC: 地点（国家、城市、地址等）
- TIME: 时间（日期、年份、时期等）
- CONCEPT: 概念（理论、方法、术语等）
- EVENT: 事件（会议、活动、项目等）
- TECH: 技术（技术、工具、平台等）
- PRODUCT: 产品（产品名称、服务等）

关系类型包括：
- 工作于 (PERSON -> ORG)
- 位于 (PERSON/ORG -> LOC)
- 包含 (ORG -> ORG)
- 使用 (PERSON/ORG -> TECH/PRODUCT)
- 参与 (PERSON -> EVENT)
- 属于 (CONCEPT -> CONCEPT)
- 发生 (EVENT -> TIME/LOC)
- 开发 (PERSON/ORG -> PRODUCT/TECH)

请以JSON格式返回结果：
{
    "entities": [
        {"name": "实体名称", "type": "实体类型", "context": "出现上下文"}
    ],
    "relations": [
        {"source": "源实体", "target": "目标实体", "relation": "关系类型", "context": "关系上下文"}
    ]
}"""),
            ("human", f"文本内容：\n{text}\n\n文档信息：{metadata}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format(text=text, metadata=metadata))
            result = json.loads(response.content)
            return result
        except Exception as e:
            logger.error(f"LLM抽取失败: {e}")
            return {'entities': [], 'relations': []}
    
    def _build_knowledge_graph(self, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """构建知识图谱"""
        # 清空现有图谱
        self.knowledge_graph.clear()
        self.entity_mapping.clear()
        self.relation_types.clear()
        
        # 添加实体节点
        for entity in entities:
            entity_id = self._normalize_entity_name(entity['name'])
            self.knowledge_graph.add_node(
                entity_id,
                name=entity['name'],
                type=entity['type'],
                context=entity.get('context', ''),
                original_name=entity['name']
            )
            self.entity_mapping[entity['name']] = entity_id
        
        # 添加关系边
        for relation in relations:
            source_id = self._normalize_entity_name(relation['source'])
            target_id = self._normalize_entity_name(relation['target'])
            
            # 确保节点存在
            if source_id not in self.knowledge_graph:
                self.knowledge_graph.add_node(source_id, name=relation['source'], type='UNKNOWN')
            if target_id not in self.knowledge_graph:
                self.knowledge_graph.add_node(target_id, name=relation['target'], type='UNKNOWN')
            
            # 添加边
            self.knowledge_graph.add_edge(
                source_id, target_id,
                relation=relation['relation'],
                context=relation.get('context', ''),
                weight=1.0
            )
            
            self.relation_types.add(relation['relation'])
        
        # 计算图谱统计信息
        stats = {
            'node_count': self.knowledge_graph.number_of_nodes(),
            'edge_count': self.knowledge_graph.number_of_edges(),
            'entity_types': list(set([node[1]['type'] for node in self.knowledge_graph.nodes(data=True)])),
            'relation_types': list(self.relation_types),
            'density': nx.density(self.knowledge_graph),
            'connected_components': nx.number_connected_components(self.knowledge_graph.to_undirected())
        }
        
        return {
            'graph': self.knowledge_graph,
            'entities': entities,
            'relations': relations,
            'statistics': stats,
            'entity_mapping': self.entity_mapping
        }
    
    def _normalize_entity_name(self, name: str) -> str:
        """标准化实体名称"""
        # 移除特殊字符，转换为小写
        normalized = re.sub(r'[^\w\s]', '', name.lower()).strip()
        # 替换空格为下划线
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized
    
    def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """获取实体的邻居节点"""
        entity_id = self._normalize_entity_name(entity_name)
        
        if entity_id not in self.knowledge_graph:
            return {'neighbors': [], 'paths': []}
        
        neighbors = []
        paths = []
        
        # 获取直接邻居
        for neighbor in self.knowledge_graph.neighbors(entity_id):
            edge_data = self.knowledge_graph.get_edge_data(entity_id, neighbor)
            neighbors.append({
                'entity': neighbor,
                'relation': edge_data['relation'],
                'context': edge_data.get('context', ''),
                'distance': 1
            })
        
        # 获取路径（最多2跳）
        if max_depth > 1:
            for target in self.knowledge_graph.nodes():
                if target != entity_id:
                    try:
                        path = nx.shortest_path(self.knowledge_graph, entity_id, target)
                        if len(path) <= max_depth + 1:
                            path_info = []
                            for i in range(len(path) - 1):
                                edge_data = self.knowledge_graph.get_edge_data(path[i], path[i + 1])
                                path_info.append({
                                    'from': path[i],
                                    'to': path[i + 1],
                                    'relation': edge_data['relation']
                                })
                            paths.append({
                                'target': target,
                                'path': path_info,
                                'length': len(path) - 1
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        return {
            'neighbors': neighbors,
            'paths': paths[:10]  # 限制路径数量
        }
    
    def search_entities(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """搜索实体"""
        results = []
        query_lower = query.lower()
        
        for node, data in self.knowledge_graph.nodes(data=True):
            if query_lower in data['name'].lower():
                if entity_type is None or data['type'] == entity_type:
                    results.append({
                        'id': node,
                        'name': data['name'],
                        'type': data['type'],
                        'context': data.get('context', ''),
                        'degree': self.knowledge_graph.degree(node)
                    })
        
        # 按度中心性排序
        results.sort(key=lambda x: x['degree'], reverse=True)
        return results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self.knowledge_graph.nodes():
            return {'error': '知识图谱为空'}
        
        # 基本统计
        stats = {
            'node_count': self.knowledge_graph.number_of_nodes(),
            'edge_count': self.knowledge_graph.number_of_edges(),
            'density': nx.density(self.knowledge_graph),
            'connected_components': nx.number_connected_components(self.knowledge_graph.to_undirected())
        }
        
        # 实体类型分布
        type_distribution = defaultdict(int)
        for node, data in self.knowledge_graph.nodes(data=True):
            type_distribution[data['type']] += 1
        stats['entity_type_distribution'] = dict(type_distribution)
        
        # 关系类型分布
        relation_distribution = defaultdict(int)
        for source, target, data in self.knowledge_graph.edges(data=True):
            relation_distribution[data['relation']] += 1
        stats['relation_type_distribution'] = dict(relation_distribution)
        
        return stats
