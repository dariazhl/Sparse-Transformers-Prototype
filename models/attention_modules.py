import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Any, Union

class SparseAttentionDemo:

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.attention_types = {
            'dense': DenseAttention(d_model, n_heads),
            'sparse_random': SparseRandomAttention(d_model, n_heads),
            'local_window': LocalWindowAttention(d_model, n_heads),
            'strided': StridedAttention(d_model, n_heads),
            'block_sparse': BlockSparseAttention(d_model, n_heads),
            'dilated': DilatedAttention(d_model, n_heads)
        }

    def compute_attention(self, attention_type: str, seq_len: int,
                         **kwargs) -> Dict[str, torch.Tensor]:
        if attention_type not in self.attention_types:
            raise ValueError(f"Unknown attention type: {attention_type}")

        x = torch.randn(1, seq_len, self.d_model)

        attention_module = self.attention_types[attention_type]
        result = attention_module.forward_with_pattern(x, **kwargs)

        return result

    def compare_attention_mechanisms(self, seq_len: int = 64) -> Dict[str, Dict[str, Any]]:
        comparison = {}

        for attention_type in self.attention_types.keys():
            try:
                result = self.compute_attention(attention_type, seq_len)

                attention_weights = result['attention_weights'][0, 0]
                sparsity = self._calculate_sparsity(attention_weights)
                connectivity = self._calculate_connectivity(attention_weights)

                comparison[attention_type] = {
                    'sparsity': sparsity,
                    'connectivity': connectivity,
                    'pattern_regularity': self._calculate_pattern_regularity(attention_weights),
                    'attention_weights': attention_weights.detach().numpy()
                }
            except Exception as e:
                comparison[attention_type] = {'error': str(e)}

        return comparison

    def _calculate_sparsity(self, attention_weights: torch.Tensor) -> float:
        total_elements = attention_weights.numel()
        zero_elements = (attention_weights == 0).sum().item()
        return zero_elements / total_elements

    def _calculate_connectivity(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        seq_len = attention_weights.shape[0]

        local_threshold = seq_len // 8
        local_connections = 0
        total_connections = (attention_weights > 1e-6).sum().item()

        for i in range(seq_len):
            for j in range(seq_len):
                if attention_weights[i, j] > 1e-6 and abs(i - j) <= local_threshold:
                    local_connections += 1

        local_ratio = local_connections / total_connections if total_connections > 0 else 0

        long_range_threshold = seq_len // 2
        long_range_connections = 0

        for i in range(seq_len):
            for j in range(seq_len):
                if attention_weights[i, j] > 1e-6 and abs(i - j) > long_range_threshold:
                    long_range_connections += 1

        long_range_ratio = long_range_connections / total_connections if total_connections > 0 else 0

        return {
            'local_ratio': local_ratio,
            'long_range_ratio': long_range_ratio,
            'total_connections': total_connections
        }

    def _calculate_pattern_regularity(self, attention_weights: torch.Tensor) -> float:
        seq_len = attention_weights.shape[0]

        row_sparsities = []
        for i in range(seq_len):
            row_sparsity = (attention_weights[i] > 1e-6).sum().item() / seq_len
            row_sparsities.append(row_sparsity)

        regularity = 1 - np.std(row_sparsities) if len(row_sparsities) > 0 else 0
        return max(0, regularity)

class BaseAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward_with_pattern(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attention_result = self.compute_attention(q, k, v, **kwargs)

        output = attention_result['output'].transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(output)

        return {
            'output': output,
            'attention_weights': attention_result['attention_weights'],
            'attention_pattern': attention_result.get('pattern', None)
        }

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class DenseAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         **kwargs) -> Dict[str, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': 'dense'
        }

class SparseRandomAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         sparsity: float = 0.8, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        num_connections = int(seq_len * seq_len * (1 - sparsity))

        for b in range(batch_size):
            for h in range(n_heads):
                indices = torch.randperm(seq_len * seq_len)[:num_connections]
                flat_mask = mask[b, h].view(-1)
                flat_mask[indices] = True

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': f'sparse_random_{sparsity}'
        }

class LocalWindowAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         window_size: int = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        if window_size is None:
            window_size = min(seq_len // 4, 64)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.zeros_like(scores, dtype=torch.bool)

        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[:, :, i, start:end] = True

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': f'local_window_{window_size}'
        }

class StridedAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         stride: int = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        if stride is None:
            stride = max(1, seq_len // 32)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.zeros_like(scores, dtype=torch.bool)

        for i in range(seq_len):
            for j in range(max(0, i-1), min(seq_len, i+2)):
                mask[:, :, i, j] = True

            for k in range(1, seq_len // stride + 1):
                if i + k * stride < seq_len:
                    mask[:, :, i, i + k * stride] = True
                if i - k * stride >= 0:
                    mask[:, :, i, i - k * stride] = True

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': f'strided_{stride}'
        }

class BlockSparseAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         block_size: int = None, sparsity: float = 0.5, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        if block_size is None:
            block_size = max(4, seq_len // 16)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        num_blocks = seq_len // block_size

        for i in range(num_blocks):
            for j in range(num_blocks):
                if torch.rand(1).item() > sparsity:
                    start_i, end_i = i * block_size, (i + 1) * block_size
                    start_j, end_j = j * block_size, (j + 1) * block_size

                    end_i = min(end_i, seq_len)
                    end_j = min(end_j, seq_len)

                    mask[:, :, start_i:end_i, start_j:end_j] = True

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': f'block_sparse_{block_size}_{sparsity}'
        }

class DilatedAttention(BaseAttention):

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         dilation_rates: List[int] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = torch.zeros_like(scores, dtype=torch.bool)

        for i in range(seq_len):
            for dilation in dilation_rates:
                for k in range(-seq_len // dilation, seq_len // dilation + 1):
                    j = i + k * dilation
                    if 0 <= j < seq_len:
                        mask[:, :, i, j] = True

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': f'dilated_{dilation_rates}'
        }

class AdaptiveSparsityAttention(BaseAttention):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads)

        self.sparsity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = q.shape

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        q_mean = q.mean(dim=1)
        k_mean = k.mean(dim=1)

        connection_probs = torch.zeros(batch_size, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                combined = torch.cat([q_mean[:, i], k_mean[:, j]], dim=-1)
                if combined.shape[-1] == self.d_model:
                    prob = self.sparsity_predictor(combined)
                    connection_probs[:, i, j] = prob.squeeze(-1)

        threshold = kwargs.get('sparsity_threshold', 0.5)
        mask = connection_probs > threshold
        mask = mask.unsqueeze(1).expand(-1, n_heads, -1, -1)

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return {
            'output': output,
            'attention_weights': attention_weights,
            'pattern': 'adaptive_sparse',
            'connection_probs': connection_probs
        }

def create_attention_comparison_matrix(seq_len: int = 32) -> Dict[str, np.ndarray]:
    demo = SparseAttentionDemo(d_model=64, n_heads=1)
    comparison = demo.compare_attention_mechanisms(seq_len)

    patterns = {}
    for attention_type, results in comparison.items():
        if 'attention_weights' in results:
            patterns[attention_type] = results['attention_weights']
        else:
            patterns[attention_type] = np.zeros((seq_len, seq_len))

    return patterns

def analyze_attention_efficiency(attention_patterns: Dict[str, np.ndarray]) -> pd.DataFrame:
    import pandas as pd

    results = []

    for pattern_name, pattern_matrix in attention_patterns.items():
        seq_len = pattern_matrix.shape[0]

        total_elements = seq_len * seq_len
        non_zero_elements = np.count_nonzero(pattern_matrix > 1e-6)
        sparsity = 1 - (non_zero_elements / total_elements)

        dense_ops = seq_len ** 2
        sparse_ops = non_zero_elements
        compute_reduction = (dense_ops - sparse_ops) / dense_ops

        memory_reduction = sparsity

        row_variations = np.std([np.count_nonzero(row > 1e-6) for row in pattern_matrix])
        regularity = 1 / (1 + row_variations)

        results.append({
            'pattern_type': pattern_name,
            'sparsity': sparsity,
            'compute_reduction': compute_reduction,
            'memory_reduction': memory_reduction,
            'regularity': regularity,
            'non_zero_elements': non_zero_elements,
            'total_elements': total_elements
        })

    return pd.DataFrame(results)

class AttentionVisualizationHelper:

    @staticmethod
    def create_attention_heatmap_data(attention_matrix: np.ndarray,
                                    title: str = "Attention Pattern") -> Dict[str, Any]:
        seq_len = attention_matrix.shape[0]

        return {
            'matrix': attention_matrix,
            'title': title,
            'x_labels': [f"Key_{i}" for i in range(seq_len)],
            'y_labels': [f"Query_{i}" for i in range(seq_len)],
            'colorscale': 'Viridis'
        }

    @staticmethod
    def analyze_attention_heads(attention_weights: torch.Tensor) -> Dict[str, Any]:
        batch_size, n_heads, seq_len, _ = attention_weights.shape

        head_analysis = {}

        for head in range(n_heads):
            head_weights = attention_weights[0, head].detach().numpy()

            sparsity = 1 - (np.count_nonzero(head_weights > 1e-6) / head_weights.size)
            entropy = -np.sum(head_weights * np.log(head_weights + 1e-12), axis=-1).mean()

            head_analysis[f'head_{head}'] = {
                'sparsity': sparsity,
                'entropy': entropy,
                'pattern': head_weights
            }

        return head_analysis

    @staticmethod
    def compare_attention_diversity(attention_patterns: Dict[str, np.ndarray]) -> Dict[str, float]:
        diversity_scores = {}

        pattern_list = list(attention_patterns.values())
        pattern_names = list(attention_patterns.keys())

        for i, (name, pattern) in enumerate(attention_patterns.items()):
            distances = []
            for j, other_pattern in enumerate(pattern_list):
                if i != j:
                    flat_pattern = pattern.flatten()
                    flat_other = other_pattern.flatten()

                    dot_product = np.dot(flat_pattern, flat_other)
                    norm_product = np.linalg.norm(flat_pattern) * np.linalg.norm(flat_other)

                    if norm_product > 0:
                        cosine_sim = dot_product / norm_product
                        cosine_dist = 1 - cosine_sim
                        distances.append(cosine_dist)

            diversity_scores[name] = np.mean(distances) if distances else 0

        return diversity_scores
