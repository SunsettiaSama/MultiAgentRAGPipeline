import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

class BGEEmbedder:
    def __init__(self, 
                 model_name='BAAI/bge-small-en', 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        """
        初始化 BGE 模型
        Args:
            model_name: BGE 模型名称（如 'BAAI/bge-small-en', 'BAAI/bge-base-zh'）
        """
        self.model = SentenceTransformer(model_name, cache_folder = './BGE_model/', device = device)
    
    def encode(self, texts, batch_size=32, show_progress=False):
        """
        将文本编码为向量
        Args:
            texts: 单个文本字符串或文本列表
            batch_size: 批处理大小（仅在 texts 是列表时生效）
            show_progress: 是否显示进度条（仅在 texts 是列表时生效）
        Returns:
            单个文本返回 1D 数组，列表返回 2D 数组（每行对应一个文本的向量）
        """
        # 处理单个文本
        if isinstance(texts, str):
            return self.model.encode([texts], convert_to_tensor=False)[0]
        # 处理文本列表
        elif isinstance(texts, list):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False
            )
            return embeddings
        else:
            raise TypeError("texts 必须是字符串或列表")
    
    def process_csv(
                self, 
                input_path, 
                output_path, 
                text_col = 'text', 
                id_col = 'id', 
                name_col = 'title', 
                batch_size=32, 
                sep = '\t', 
                **kwargs, 
                   ):
        """
        读取 CSV 文件，生成嵌入并保存
        Args:
            input_path: 输入 CSV 文件路径（含 [文章id, 文章名, 文章] 列）
            output_path: 输出 CSV 文件路径（含 [文章id, 文章名, embedding] 列）
            text_col: 文本列名（默认 '文章'）
            id_col: 文章ID列名（默认 '文章id'）
            name_col: 文章名列名（默认 '文章名'）
            batch_size: 批处理大小（控制内存使用）
        """
        print('=' * 40)
        print('Reading csv...')
        # 读取 CSV 文件
        df = pd.read_csv(input_path, sep = '\t', encoding_errors = 'ignore', on_bad_lines = 'skip', engine = 'python')
        required_cols = [id_col, name_col, text_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 必须包含列：{required_cols}")
        
        # 提取文本和元数据
        texts = df[text_col].tolist()
        metadata = df[[id_col, name_col]].copy()
        
        # 生成嵌入（分批次处理）
        embeddings = []
        print('=' * 20)
        print('Start Processing csv to embedding csv...')
        print('=' * 20)

        with tqdm(total = len(texts) / batch_size + 1) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.encode(batch_texts)
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        
        # 将嵌入转换为字符串格式（CSV 可存储）
        embedding_strings = [np.array2string(vec, separator=',') for vec in embeddings]
        
        # 合并元数据和嵌入
        metadata['embedding'] = embedding_strings

        print('=' * 20)
        # 保存结果
        metadata.to_csv(output_path, index=False)
        print(f"成功保存 {len(metadata)} 条记录到 {output_path}")
