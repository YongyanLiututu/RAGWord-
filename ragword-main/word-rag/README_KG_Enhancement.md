# 知识图谱增强RAG系统

## 概述

本系统在原有RAG_Word项目基础上，新增了知识图谱（Knowledge Graph, KG）增强模块，通过实体关系抽取、图数据库检索和一致性验证，显著提升了检索精度和答案质量。

## 新增功能

### 1. 知识图谱加载器 (`kg_loader.py`)

**功能**：从用户上传文档中抽取实体和关系，构建局部知识图谱

**主要特性**：
- 支持8种实体类型：人物、机构、地点、时间、概念、事件、技术、产品
- 支持多种关系类型：工作于、位于、包含、使用、参与、属于、发生、开发
- 使用LLM进行语义实体关系抽取
- 基于NetworkX构建图数据结构
- 提供实体标准化和去重功能

**使用示例**：
```python
from kg_loader import KnowledgeGraphLoader
from langchain.schema import Document

# 初始化KG加载器
kg_loader = KnowledgeGraphLoader(model_name="gpt-4")

# 从文档中抽取实体和关系
documents = [Document(page_content="OpenAI是一家AI公司，由Sam Altman创立。")]
kg_data = kg_loader.extract_entities_and_relations(documents)

print(f"抽取了 {kg_data['statistics']['node_count']} 个实体")
print(f"抽取了 {kg_data['statistics']['edge_count']} 个关系")
```

### 2. 知识图谱检索器 (`kg_retriever.py`)

**功能**：基于图数据库实现KG查询，支持节点+关系匹配

**主要特性**：
- 实体索引和关系索引
- 语义查询意图解析
- 实体相关度计算
- 路径查询和邻居查询
- 支持多种查询模式

**使用示例**：
```python
from kg_retriever import KnowledgeGraphRetriever

# 初始化KG检索器
kg_retriever = KnowledgeGraphRetriever(
    knowledge_graph=kg_data['graph'],
    model_name="gpt-4"
)

# 实体查询
entities = kg_retriever.query_entities("OpenAI")

# 语义查询
result = kg_retriever.semantic_query("OpenAI的CEO是谁？")
```

### 3. KG增强混合检索器 (`hybrid_retriever.py`)

**功能**：在原有FAISS+BM25基础上增加KG通道，实现三通道融合检索

**主要特性**：
- 保持原有向量检索和关键词检索功能
- 新增KG检索通道
- 支持可配置的KG权重
- 智能结果融合算法

**使用示例**：
```python
from hybrid_retriever import KGEnhancedHybridRetriever

# 初始化KG增强检索器
retriever = KGEnhancedHybridRetriever(
    embeddings=embeddings,
    llm_model=chat_model,
    kg_retriever=kg_retriever,
    enable_kg=True,
    kg_weight=0.3
)

# KG增强检索
results = retriever.kg_enhanced_retrieve("查询内容")
```

### 4. KG感知Agent (`enhanced_agents.py`)

**功能**：专门处理KG检索和验证的智能Agent

**主要特性**：
- 自动构建知识图谱
- KG增强检索
- 查询意图分析
- KG结果验证
- 实体搜索功能

**使用示例**：
```python
from enhanced_agents import KGAwareAgent

# 初始化KG Agent
kg_agent = KGAwareAgent(
    kg_loader=kg_loader,
    kg_retriever=kg_retriever,
    model_name="gpt-4"
)

# 构建知识图谱
build_result = kg_agent.build_knowledge_graph(documents)

# KG增强检索
retrieval_result = kg_agent.kg_enhanced_retrieval("查询内容")
```

### 5. KG增强生成器 (`layered_generator.py`)

**功能**：在分层生成基础上增加KG一致性检查

**主要特性**：
- 保持原有分层生成逻辑
- 新增KG一致性验证
- 基于KG验证结果改进答案
- 实体提取和验证

**使用示例**：
```python
from layered_generator import KGEnhancedLayeredGenerator

# 初始化KG增强生成器
generator = KGEnhancedLayeredGenerator(
    model_name="gpt-4",
    kg_loader=kg_loader,
    enable_kg_verification=True
)

# KG增强答案生成
result = generator.generate_layered_answer_with_kg_verification(
    question="问题",
    documents=documents,
    kg_info=kg_data
)
```

## 系统集成

### 增强版RAG系统 (`enhanced_rag_system.py`)

**新增配置参数**：
```python
rag_system = EnhancedRAGSystem(
    model_name="gpt-4",
    emb_model="text-embedding-ada-002",
    use_mistral=False,
    enable_enhanced_features=True,
    enable_kg=True,           # 启用KG功能
    kg_mode="fusion"          # KG模式：fusion/verify
)
```

**新增方法**：
- `get_kg_statistics()`: 获取KG统计信息
- `search_entity_in_kg()`: 在KG中搜索实体
- `kg_enhanced_query()`: KG增强查询
- `set_kg_config()`: 设置KG配置
- `export_knowledge_graph()`: 导出知识图谱

**使用流程**：
```python
# 1. 创建系统实例
rag_system = EnhancedRAGSystem(enable_kg=True)

# 2. 上传文档（自动构建KG）
result = rag_system.upload_documents("word")
print(f"KG构建结果: {result['kg_result']}")

# 3. 问答（自动使用KG增强）
response = rag_system.ask_question("问题内容")
print(f"答案: {response['answer']}")

# 4. KG相关查询
kg_stats = rag_system.get_kg_statistics()
kg_query = rag_system.kg_enhanced_query("KG查询")
```

## 配置选项

### KG功能开关

```python
# 完全禁用KG功能
rag_system = EnhancedRAGSystem(enable_kg=False)

# 启用KG功能
rag_system = EnhancedRAGSystem(enable_kg=True)
```

### KG模式配置

```python
# fusion模式：KG与RAG融合
rag_system = EnhancedRAGSystem(kg_mode="fusion")

# verify模式：KG仅用于验证
rag_system = EnhancedRAGSystem(kg_mode="verify")
```

### KG权重调整

```python
# 调整KG结果权重（0.0-1.0）
rag_system.set_kg_config(kg_weight=0.5)
```

## 性能优化

### 1. 实体抽取优化
- 使用LLM进行语义抽取，提高准确性
- 支持实体标准化和去重
- 可配置的实体类型和关系类型

### 2. 检索性能优化
- 构建实体和关系索引
- 支持语义查询意图解析
- 智能结果融合算法

### 3. 生成质量优化
- KG一致性检查
- 基于KG验证的答案改进
- 实体关系验证

## 测试

运行测试脚本验证KG功能：

```bash
cd ragword-main/word-rag
python test_kg_system.py
```

测试内容包括：
- KG加载器功能测试
- KG检索器功能测试
- KG Agent功能测试
- 增强RAG系统集成测试
- KG增强生成器测试

## 依赖要求

### 新增依赖
```bash
pip install networkx
pip install spacy  # 可选，用于实体识别
```

### 环境变量
```bash
# OpenAI API密钥
export OPENAI_API_KEY="your_openai_api_key"

# MistralAI API密钥（可选）
export MISTRAL_API_KEY="your_mistral_api_key"
```

## 注意事项

1. **保持向后兼容**：所有新增功能都是可选的，不影响原有功能
2. **性能考虑**：KG构建需要额外的LLM调用，建议在文档上传时进行
3. **内存使用**：大型知识图谱可能占用较多内存
4. **API限制**：KG功能依赖LLM API，注意调用频率限制

## 故障排除

### 常见问题

1. **KG加载器初始化失败**
   - 检查API密钥配置
   - 确认网络连接正常

2. **KG构建失败**
   - 检查文档内容是否包含实体信息
   - 查看日志中的具体错误信息

3. **KG检索无结果**
   - 确认知识图谱已正确构建
   - 检查查询内容是否与KG中的实体相关

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 未来扩展

1. **图数据库支持**：集成Neo4j等专业图数据库
2. **可视化界面**：添加KG可视化功能
3. **增量更新**：支持KG的增量构建和更新
4. **多语言支持**：扩展多语言实体关系抽取
5. **知识推理**：增加基于KG的逻辑推理功能
