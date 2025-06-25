import numpy as np
import torch
from typing import Tuple, Dict, Any

def generate_attention_matrix(seq_len: int, pattern_type: str = "dense", sparsity: float = 0.0, **kwargs) -> np.ndarray:
    if pattern_type == "dense":
        matrix = np.random.uniform(0.1, 1.0, (seq_len, seq_len))
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                matrix[i, j] *= 0.3

    elif pattern_type == "sparse":
        matrix = np.zeros((seq_len, seq_len))
        num_connections = int(seq_len * seq_len * (1 - sparsity))

        indices = np.random.choice(seq_len * seq_len, num_connections, replace=False)
        for idx in indices:
            i, j = divmod(idx, seq_len)
            matrix[i, j] = np.random.uniform(0.1, 1.0)

    elif pattern_type == "local":
        window_size = kwargs.get('window_size', max(1, seq_len // 8))
        matrix = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.5, 1.0)

    elif pattern_type == "strided":
        stride = kwargs.get('stride', max(1, seq_len // 16))
        matrix = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                matrix[i, j] = np.random.uniform(0.3, 1.0)

            for k in range(1, seq_len // stride):
                j = i + k * stride
                if j < seq_len:
                    matrix[i, j] = np.random.uniform(0.2, 0.8)
                j = i - k * stride
                if j >= 0:
                    matrix[i, j] = np.random.uniform(0.2, 0.8)

    elif pattern_type == "random":
        matrix = np.random.uniform(0, 1, (seq_len, seq_len))
        mask = np.random.random((seq_len, seq_len)) < (1 - sparsity)
        matrix = matrix * mask

    elif pattern_type == "structured":
        block_size = kwargs.get('block_size', max(1, seq_len // 8))
        matrix = np.zeros((seq_len, seq_len))

        num_blocks = seq_len // block_size
        for i in range(num_blocks):
            for j in range(num_blocks):
                if np.random.random() > sparsity:
                    start_i, end_i = i * block_size, (i + 1) * block_size
                    start_j, end_j = j * block_size, (j + 1) * block_size
                    matrix[start_i:end_i, start_j:end_j] = np.random.uniform(0.1, 1.0, (block_size, block_size))

    elif pattern_type == "custom":
        window_size = kwargs.get('window_size', seq_len // 8)
        stride = kwargs.get('stride', seq_len // 16)
        random_ratio = kwargs.get('random_ratio', 0.1)

        matrix = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.4, 1.0)

        for i in range(seq_len):
            for k in range(1, seq_len // stride):
                j = i + k * stride
                if j < seq_len:
                    matrix[i, j] = max(matrix[i, j], np.random.uniform(0.2, 0.6))

        num_random = int(seq_len * seq_len * random_ratio)
        indices = np.random.choice(seq_len * seq_len, num_random, replace=False)
        for idx in indices:
            i, j = divmod(idx, seq_len)
            matrix[i, j] = max(matrix[i, j], np.random.uniform(0.1, 0.5))

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums

    return matrix

def calculate_sparsity_metrics(attention_matrix: np.ndarray) -> Dict[str, Any]:
    seq_len = attention_matrix.shape[0]
    total_elements = seq_len * seq_len

    non_zero_elements = np.count_nonzero(attention_matrix)
    sparsity_ratio = 1 - (non_zero_elements / total_elements)

    memory_mb = (non_zero_elements * 4) / (1024 * 1024)

    effective_complexity = non_zero_elements

    local_window = max(1, seq_len // 16)
    local_connections = 0
    for i in range(seq_len):
        for j in range(max(0, i - local_window), min(seq_len, i + local_window + 1)):
            if attention_matrix[i, j] > 0:
                local_connections += 1

    local_ratio = local_connections / non_zero_elements if non_zero_elements > 0 else 0

    long_range_connections = 0
    long_range_threshold = seq_len // 4
    for i in range(seq_len):
        for j in range(seq_len):
            if attention_matrix[i, j] > 0 and abs(i - j) > long_range_threshold:
                long_range_connections += 1

    long_range_ratio = long_range_connections / non_zero_elements if non_zero_elements > 0 else 0

    row_sparsities = [np.count_nonzero(row) / seq_len for row in attention_matrix]
    regularity_score = 1 - np.std(row_sparsities) if len(row_sparsities) > 0 else 0

    return {
        "non_zero_elements": non_zero_elements,
        "sparsity_ratio": sparsity_ratio,
        "memory_mb": memory_mb,
        "effective_complexity": effective_complexity,
        "local_ratio": local_ratio,
        "long_range_ratio": long_range_ratio,
        "regularity_score": regularity_score
    }

def compute_attention_efficiency(seq_len: int, sparsity: float) -> Dict[str, float]:
    dense_ops = seq_len ** 2
    sparse_ops = dense_ops * (1 - sparsity)

    dense_memory = seq_len ** 2 * 4
    sparse_memory = sparse_ops * 4
    memory_reduction = dense_memory / sparse_memory if sparse_memory > 0 else float('inf')

    compute_reduction = dense_ops / sparse_ops if sparse_ops > 0 else float('inf')

    theoretical_speedup = min(memory_reduction, compute_reduction) * 0.8

    return {
        "memory_reduction": memory_reduction,
        "compute_reduction": compute_reduction,
        "theoretical_speedup": theoretical_speedup,
        "dense_ops": dense_ops,
        "sparse_ops": sparse_ops
    }

def visualize_attention_pattern(attention_matrix: np.ndarray, title: str = "Attention Pattern") -> Dict[str, Any]:
    seq_len = attention_matrix.shape[0]

    metrics = calculate_sparsity_metrics(attention_matrix)

    heatmap_data = {
        "matrix": attention_matrix,
        "x_labels": [f"Pos_{i}" for i in range(seq_len)],
        "y_labels": [f"Pos_{i}" for i in range(seq_len)],
        "title": title
    }

    attention_stats = {
        "position": list(range(seq_len)),
        "total_attention": attention_matrix.sum(axis=1).tolist(),
        "num_connections": [np.count_nonzero(row) for row in attention_matrix],
        "max_attention": attention_matrix.max(axis=1).tolist(),
        "mean_attention": attention_matrix.mean(axis=1).tolist()
    }

    return {
        "heatmap_data": heatmap_data,
        "metrics": metrics,
        "attention_stats": attention_stats
    }

class AttentionPatternGenerator:

    def __init__(self):
        self.patterns = {
            "dense": self._generate_dense,
            "sparse": self._generate_sparse,
            "local": self._generate_local,
            "strided": self._generate_strided,
            "block": self._generate_block,
            "random": self._generate_random
        }

    def generate(self, pattern_type: str, seq_len: int, **kwargs) -> np.ndarray:
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        return self.patterns[pattern_type](seq_len, **kwargs)

    def _generate_dense(self, seq_len: int, **kwargs) -> np.ndarray:
        return generate_attention_matrix(seq_len, "dense", 0.0, **kwargs)

    def _generate_sparse(self, seq_len: int, sparsity: float = 0.8, **kwargs) -> np.ndarray:
        return generate_attention_matrix(seq_len, "sparse", sparsity, **kwargs)

    def _generate_local(self, seq_len: int, window_size: int = None, **kwargs) -> np.ndarray:
        if window_size is None:
            window_size = max(1, seq_len // 8)
        return generate_attention_matrix(seq_len, "local", 0.0, window_size=window_size, **kwargs)

    def _generate_strided(self, seq_len: int, stride: int = None, **kwargs) -> np.ndarray:
        if stride is None:
            stride = max(1, seq_len // 16)
        return generate_attention_matrix(seq_len, "strided", 0.0, stride=stride, **kwargs)

    def _generate_block(self, seq_len: int, block_size: int = None, sparsity: float = 0.5, **kwargs) -> np.ndarray:
        if block_size is None:
            block_size = max(1, seq_len // 8)
        return generate_attention_matrix(seq_len, "structured", sparsity, block_size=block_size, **kwargs)

    def _generate_random(self, seq_len: int, sparsity: float = 0.7, **kwargs) -> np.ndarray:
        return generate_attention_matrix(seq_len, "random", sparsity, **kwargs)
