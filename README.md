# PDF RAG 问答系统

基于 RAG（检索增强生成）技术的 PDF 文档智能问答系统。上传任意 PDF 文件，即可对文档内容进行自然语言提问。

## 技术栈

- **后端框架**：FastAPI
- **向量数据库**：Chroma
- **PDF 处理**：PyMuPDF
- **Embedding 模型**：智谱 AI（embedding-3）
- **对话模型**：DeepSeek（deepseek-chat）

## 功能

- 上传 PDF 文件，自动切片、转向量、存入向量数据库
- 基于文档内容进行自然语言问答
- 检索最相关内容后由大模型生成准确回答

## 接口说明

- `POST /upload-and-index/`：上传 PDF，建立知识库
- `POST /ask/`：提问，返回基于文档的回答

## 运行方法

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量，新建 `.env` 文件：

   ```
   API_KEY=你的DeepSeek API Key
   ZHIPU_API_KEY=你的智谱AI API Key
   ```

3. 启动服务
```bash
uvicorn main:app --reload
```

4. 访问 `http://127.0.0.1:8000/docs` 测试接口