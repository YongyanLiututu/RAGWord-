#!/usr/bin/env python3
"""
知识图谱增强RAG系统测试脚本
"""

import os
import sys
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_kg_loader():
    """测试知识图谱加载器"""
    logger.info("=== 测试知识图谱加载器 ===")
    
    try:
        from kg_loader import KnowledgeGraphLoader
        from langchain.schema import Document
        
        # 创建测试文档
        test_documents = [
            Document(
                page_content="OpenAI是一家专注于人工智能研究的公司，由Sam Altman创立。GPT-4是他们开发的大型语言模型。",
                metadata={"source": "test_doc_1"}
            ),
            Document(
                page_content="Sam Altman是OpenAI的CEO，他领导了GPT系列模型的开发。",
                metadata={"source": "test_doc_2"}
            ),
            Document(
                page_content="GPT-4是一个多模态大型语言模型，能够处理文本和图像输入。",
                metadata={"source": "test_doc_3"}
            )
        ]
        
        # 初始化KG加载器
        kg_loader = KnowledgeGraphLoader(model_name="gpt-4", use_mistral=False)
        
        # 抽取实体和关系
        kg_data = kg_loader.extract_entities_and_relations(test_documents)
        
        logger.info(f"KG构建结果: {kg_data['statistics']}")
        logger.info(f"实体数量: {len(kg_data['entities'])}")
        logger.info(f"关系数量: {len(kg_data['relations'])}")
        
        return kg_data
        
    except Exception as e:
        logger.error(f"KG加载器测试失败: {e}")
        return None

def test_kg_retriever(kg_data):
    """测试知识图谱检索器"""
    logger.info("=== 测试知识图谱检索器 ===")
    
    try:
        from kg_retriever import KnowledgeGraphRetriever
        
        if not kg_data or 'graph' not in kg_data:
            logger.warning("KG数据不可用，跳过检索器测试")
            return None
        
        # 初始化KG检索器
        kg_retriever = KnowledgeGraphRetriever(
            knowledge_graph=kg_data['graph'],
            model_name="gpt-4",
            use_mistral=False
        )
        
        # 测试实体查询
        entities = kg_retriever.query_entities("OpenAI")
        logger.info(f"实体查询结果: {len(entities)} 个实体")
        
        # 测试语义查询
        semantic_result = kg_retriever.semantic_query("OpenAI的CEO是谁？")
        logger.info(f"语义查询结果: {semantic_result}")
        
        return kg_retriever
        
    except Exception as e:
        logger.error(f"KG检索器测试失败: {e}")
        return None

def test_kg_agent(kg_loader, kg_retriever):
    """测试KG感知Agent"""
    logger.info("=== 测试KG感知Agent ===")
    
    try:
        from enhanced_agents import KGAwareAgent
        from langchain.schema import Document
        
        # 创建测试文档
        test_documents = [
            Document(
                page_content="Microsoft是一家科技公司，开发了Windows操作系统。",
                metadata={"source": "test_doc_4"}
            )
        ]
        
        # 初始化KG Agent
        kg_agent = KGAwareAgent(
            kg_loader=kg_loader,
            kg_retriever=kg_retriever,
            model_name="gpt-4",
            use_mistral=False
        )
        
        # 测试KG构建
        build_result = kg_agent.build_knowledge_graph(test_documents)
        logger.info(f"KG构建结果: {build_result['status']}")
        
        # 测试KG检索
        retrieval_result = kg_agent.kg_enhanced_retrieval("Microsoft")
        logger.info(f"KG检索结果: {retrieval_result['status']}")
        
        return kg_agent
        
    except Exception as e:
        logger.error(f"KG Agent测试失败: {e}")
        return None

def test_enhanced_rag_system():
    """测试增强版RAG系统"""
    logger.info("=== 测试增强版RAG系统 ===")
    
    try:
        from enhanced_rag_system import EnhancedRAGSystem
        
        # 创建系统实例（启用KG功能）
        rag_system = EnhancedRAGSystem(
            model_name="gpt-4",
            emb_model="text-embedding-ada-002",
            use_mistral=False,
            enable_enhanced_features=True,
            enable_kg=True,
            kg_mode="fusion"
        )
        
        # 获取系统状态
        status = rag_system.get_system_status()
        logger.info(f"系统状态: {status}")
        
        # 测试KG统计
        kg_stats = rag_system.get_kg_statistics()
        logger.info(f"KG统计: {kg_stats}")
        
        return rag_system
        
    except Exception as e:
        logger.error(f"增强RAG系统测试失败: {e}")
        return None

def test_kg_generator():
    """测试KG增强生成器"""
    logger.info("=== 测试KG增强生成器 ===")
    
    try:
        from layered_generator import KGEnhancedLayeredGenerator
        from kg_loader import KnowledgeGraphLoader
        from langchain.schema import Document
        
        # 创建测试文档
        test_documents = [
            Document(
                page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                metadata={"source": "test_doc_5"}
            )
        ]
        
        # 初始化KG加载器
        kg_loader = KnowledgeGraphLoader(model_name="gpt-4", use_mistral=False)
        
        # 初始化KG增强生成器
        generator = KGEnhancedLayeredGenerator(
            model_name="gpt-4",
            kg_loader=kg_loader,
            enable_kg_verification=True
        )
        
        # 构建KG数据
        kg_data = kg_loader.extract_entities_and_relations(test_documents)
        
        # 测试KG增强生成
        result = generator.generate_layered_answer_with_kg_verification(
            "什么是人工智能？",
            test_documents,
            kg_data
        )
        
        logger.info(f"KG增强生成结果: {result['generation_method']}")
        logger.info(f"KG验证应用: {result['kg_enhancement_applied']}")
        
        return generator
        
    except Exception as e:
        logger.error(f"KG增强生成器测试失败: {e}")
        return None

def main():
    """主测试函数"""
    logger.info("开始知识图谱增强RAG系统测试")
    
    # 测试KG加载器
    kg_data = test_kg_loader()
    
    # 测试KG检索器
    kg_retriever = test_kg_retriever(kg_data)
    
    # 测试KG Agent
    kg_agent = test_kg_agent(kg_data['kg_loader'] if kg_data else None, kg_retriever)
    
    # 测试增强RAG系统
    rag_system = test_enhanced_rag_system()
    
    # 测试KG增强生成器
    generator = test_kg_generator()
    
    logger.info("=== 测试总结 ===")
    logger.info(f"KG加载器: {'✓' if kg_data else '✗'}")
    logger.info(f"KG检索器: {'✓' if kg_retriever else '✗'}")
    logger.info(f"KG Agent: {'✓' if kg_agent else '✗'}")
    logger.info(f"增强RAG系统: {'✓' if rag_system else '✗'}")
    logger.info(f"KG增强生成器: {'✓' if generator else '✗'}")
    
    logger.info("知识图谱增强RAG系统测试完成")

if __name__ == "__main__":
    main()
