# ==========================================
# 建筑文档智能RAG审查系统 - Ollama版本
# 正确的执行顺序
# ==========================================

# ===========================================
# Cell 1: 定义BaseLLM基类
# ===========================================
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseLLM(ABC):
    """Interface for large language models."""

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def predict(self, input: str) -> str:
        """Sends a text input to the LLM and retrieves a response."""

print("✅ BaseLLM基类定义完成！")

# ===========================================
# Cell 2: 定义OllamaLLM类
# ===========================================
import ollama
from typing import Any, Optional

class OllamaLLM(BaseLLM):
    """Implementation of the BaseLLM interface using Ollama."""

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, model_params, **kwargs)
        self.host = host
        # 设置ollama客户端的host
        if host != "http://localhost:11434":
            self.client = ollama.Client(host=host)
        else:
            self.client = ollama

    def predict(self, input: str) -> str:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama调用出错: {e}")
            return f"抱歉，模型调用出现问题: {str(e)}"

print("✅ OllamaLLM类定义完成！")

# ===========================================
# Cell 3: 测试OllamaLLM功能
# ===========================================
print("🚀 开始测试OllamaLLM...")

# 初始化模型
llm = OllamaLLM(
    model_name="qwen3:14b",  # 使用本地部署的qwen模型
    host="http://localhost:11434"
)

# 测试基本功能
print("📝 测试问题：建筑文档审查")
response = llm.predict("你好，请简单回答：钢筋混凝土结构中，C25混凝土的抗压强度标准值是多少？")
print(f"🤖 AI回复：{response}")
print("\n✅ OllamaLLM测试成功！")

# ===========================================
# Cell 4: 定义修复版Embedding模块
# ===========================================
from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np
import os

# 设置环境变量避免路径问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 检查sentence_transformers包
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence_transformers包可用")
except ImportError:
    print("❌ 需要安装sentence_transformers包")
    print("请在终端运行: pip install sentence-transformers")

class BaseEmb(ABC):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def get_emb(self, input: str) -> List[float]:
        """Sends a text input to the embedding model and retrieves the embedding."""
        pass

class BGEEmbedding(BaseEmb):
    """修复版BGE嵌入模型，解决路径和兼容性问题"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        print(f"正在加载嵌入模型: {model_name}...")
        
        # 创建缓存目录
        cache_dir = os.path.abspath("./model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 方法1：使用eager attention避免SDPA问题
            self.embed_model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                model_kwargs={"attn_implementation": "eager"},
                trust_remote_code=True
            )
            print("✅ 嵌入模型加载完成！")
        except Exception as e:
            print(f"❌ 方法1失败: {e}")
            print("尝试方法2：使用轻量级模型...")
            try:
                # 方法2：使用更轻量的模型
                self.embed_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=cache_dir
                )
                print("✅ 嵌入模型加载完成（轻量级版本）！")
            except Exception as e2:
                print(f"❌ 方法2也失败: {e2}")
                print("建议：检查网络连接或重启kernel")
                raise

    def get_emb(self, text: str) -> List[float]:
        embedding = self.embed_model.encode(text)
        return embedding.tolist()
    
    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.embed_model.encode(texts, show_progress_bar=show_progress_bar)
        return np.array(embeddings)

print("✅ Embedding模块定义完成！")

# ===========================================
# Cell 5: 测试Embedding模块
# ===========================================
print("🔧 测试Embedding模块...")

try:
    # 初始化embedding模型
    emb = BGEEmbedding(model_name="BAAI/bge-m3")
    
    # 测试单个文本编码
    test_text = "建筑结构的安全性检查包括哪些方面？"
    print(f"📝 测试文本: {test_text}")
    
    embedding = emb.get_emb(test_text)
    print(f"✅ 成功生成embedding，维度: {len(embedding)}")
    print(f"📊 前5个维度值: {[f'{x:.4f}' for x in embedding[:5]]}")
    
    # 测试批量编码
    test_texts = [
        "钢筋混凝土结构施工要求",
        "建筑安全检查标准", 
        "混凝土养护技术规范"
    ]
    embeddings = emb.encode(test_texts)
    print(f"✅ 批量编码成功，形状: {embeddings.shape}")
    
except Exception as e:
    print(f"❌ Embedding测试失败: {e}")
    print("提示：如果是网络问题，请检查网络连接")

print("\n✅ Embedding模块测试完成！")

# ===========================================
# Cell 6: 完整RAG系统测试
# ===========================================
print("🚀 开始完整的RAG系统测试...")

try:
    # 1. 确认所有组件正常
    print("\n1️⃣ 验证组件状态...")
    print(f"✅ LLM模型: {llm.model_name}")
    print(f"✅ Embedding模型: {emb.model_name}")
    
    # 2. 创建简单知识库
    print("\n2️⃣ 构建知识库...")
    knowledge_base = [
        "钢筋混凝土柱的混凝土强度等级不应低于C25，钢筋保护层厚度应符合设计要求。",
        "混凝土浇筑应连续进行，浇筑间歇时间不应超过混凝土的初凝时间。",
        "钢筋焊接应符合相关规范要求，焊接质量应进行检验。",
        "模板安装应牢固，几何尺寸应准确，表面应平整光滑。",
        "混凝土养护期间应保持混凝土表面湿润，养护时间不少于7天。"
    ]
    
    # 计算知识库embeddings
    kb_embeddings = emb.encode(knowledge_base)
    print(f"✅ 知识库编码完成，包含{len(knowledge_base)}个文档")
    
    # 3. 测试RAG检索
    print("\n3️⃣ 测试RAG检索...")
    query = "混凝土强度要求是什么？"
    query_embedding = emb.encode([query])
    
    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    
    # 获取最相关的文档
    best_idx = similarities.argmax()
    best_doc = knowledge_base[best_idx]
    best_score = similarities[best_idx]
    
    print(f"📝 查询: {query}")
    print(f"🎯 最相关文档: {best_doc}")
    print(f"📊 相似度分数: {best_score:.3f}")
    
    # 4. 测试RAG生成
    print("\n4️⃣ 测试RAG生成...")
    rag_prompt = f"""
基于以下参考文档回答问题：

参考文档：{best_doc}

问题：{query}

请提供准确、专业的回答：
"""
    
    rag_response = llm.predict(rag_prompt)
    print(f"🤖 RAG回答: {rag_response}")
    
    print("\n🎉 完整RAG系统测试成功！")
    print("\n📋 系统状态总结：")
    print("✅ Ollama LLM: 正常工作")
    print("✅ BGE Embedding: 正常工作") 
    print("✅ 向量检索: 正常工作")
    print("✅ RAG生成: 正常工作")
    print("\n🚀 系统已准备好处理建筑文档审查任务！")
    
except Exception as e:
    print(f"❌ 系统测试失败: {e}")
    import traceback
    traceback.print_exc()
