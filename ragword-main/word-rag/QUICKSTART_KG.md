# 知识图谱增强RAG系统 - 快速开始指南

## 1. 环境准备

### 安装依赖
```bash
# 安装基础依赖
pip install -r requirements_kg.txt

# 可选：安装spaCy模型（用于实体识别）
python -m spacy download zh_core_web_sm  # 中文
python -m spacy download en_core_web_sm  # 英文
```

### 配置API密钥
```bash
# 创建.env文件
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "MISTRAL_API_KEY=your_mistral_api_key_here" >> .env  # 可选
```

## 2. 基础使用

### 创建系统实例
```python
from enhanced_rag_system import EnhancedRAGSystem

# 创建启用KG功能的系统
rag_system = EnhancedRAGSystem(
    model_name="gpt-4",
    emb_model="text-embedding-ada-002",
    enable_enhanced_features=True,
    enable_kg=True,           # 启用KG功能
    kg_mode="fusion"          # KG与RAG融合模式
)
```

### 上传文档
```python
# 上传Word文档（自动构建知识图谱）
result = rag_system.upload_documents("path/to/your/document.docx")

# 查看KG构建结果
if result['kg_result']:
    print(f"KG构建成功：{result['kg_result']['statistics']}")
```

### 问答交互
```python
# 普通问答（自动使用KG增强）
response = rag_system.ask_question("什么是人工智能？")
print(f"答案：{response['answer']}")

# KG增强查询
kg_query = rag_system.kg_enhanced_query("OpenAI的相关技术")
print(f"KG查询结果：{kg_query}")
```

## 3. 高级功能

### KG统计信息
```python
# 获取KG统计
kg_stats = rag_system.get_kg_statistics()
print(f"实体数量：{kg_stats['total_entities']}")
print(f"关系数量：{kg_stats['total_relations']}")
```

### 实体搜索
```python
# 在KG中搜索特定实体
search_result = rag_system.search_entity_in_kg("OpenAI", entity_type="ORG")
print(f"搜索结果：{search_result}")
```

### 配置调整
```python
# 调整KG权重
rag_system.set_kg_config(kg_weight=0.5)

# 导出知识图谱
kg_export = rag_system.export_knowledge_graph(format='json')
```

## 4. 测试验证

### 运行测试
```bash
python test_kg_system.py
```

### 检查系统状态
```python
status = rag_system.get_system_status()
print(f"KG功能状态：{status['knowledge_graph']}")
```

## 5. 常见用例

### 用例1：技术文档问答
```python
# 上传技术文档
rag_system.upload_documents("technical_docs.docx")

# 技术相关问题
response = rag_system.ask_question("这个技术的主要特点是什么？")
response = rag_system.ask_question("谁开发了这个技术？")
```

### 用例2：公司信息查询
```python
# 上传公司相关文档
rag_system.upload_documents("company_info.docx")

# 公司信息查询
response = rag_system.ask_question("公司的CEO是谁？")
response = rag_system.ask_question("公司的主要产品有哪些？")
```

### 用例3：学术论文分析
```python
# 上传学术论文
rag_system.upload_documents("research_paper.docx")

# 学术问题
response = rag_system.ask_question("论文的主要贡献是什么？")
response = rag_system.ask_question("使用了哪些研究方法？")
```

## 6. 性能优化建议

### 文档预处理
- 确保文档包含丰富的实体信息
- 文档结构清晰，便于实体抽取
- 避免过于复杂的格式

### 系统配置
- 根据文档规模调整KG权重
- 合理设置检索数量参数
- 监控API调用频率

### 内存管理
- 大型知识图谱考虑分批处理
- 定期清理不需要的缓存
- 监控系统内存使用

## 7. 故障排除

### 问题1：KG构建失败
```python
# 检查API密钥
import os
print(f"OpenAI API Key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")

# 检查文档内容
print(f"文档数量: {len(documents)}")
print(f"文档内容示例: {documents[0].page_content[:100]}")
```

### 问题2：KG检索无结果
```python
# 检查KG是否构建成功
kg_stats = rag_system.get_kg_statistics()
print(f"KG实体数量: {kg_stats.get('total_entities', 0)}")

# 检查查询内容
print(f"查询内容: {query}")
```

### 问题3：系统性能问题
```python
# 检查系统状态
status = rag_system.get_system_status()
print(f"系统状态: {status}")

# 调整配置
rag_system.set_kg_config(kg_weight=0.2)  # 降低KG权重
```

## 8. 下一步

1. **阅读详细文档**：查看 `README_KG_Enhancement.md`
2. **运行完整测试**：执行 `test_kg_system.py`
3. **探索高级功能**：尝试不同的KG配置和模式
4. **集成到应用**：将KG增强功能集成到您的应用中

## 支持

如果遇到问题，请：
1. 查看日志输出
2. 检查API密钥配置
3. 确认依赖安装完整
4. 参考故障排除部分
