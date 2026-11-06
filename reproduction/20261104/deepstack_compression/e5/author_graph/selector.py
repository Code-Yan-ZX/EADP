"""Unchanged author-op graph adapter. No command-line GPU entry or authority."""
import torch


def validate(importance, similarity, k):
    if (not torch.is_tensor(importance) or not torch.is_tensor(similarity)
        or importance.ndim != 2 or similarity.ndim != 3
        or importance.shape[0] != 1 or similarity.shape != (1, importance.shape[1], importance.shape[1])
        or type(k) is not int or not 1 <= k <= importance.shape[1]
        or importance.device != similarity.device or importance.device.type != 'cuda'
        or importance.dtype != torch.float32 or similarity.dtype != torch.float32
        or not importance.is_contiguous() or not similarity.is_contiguous()):
        raise ValueError('expected contiguous batch-one float32 CUDA selector inputs and valid K')


class AuthorGraph:
    """Single stream, fixed shape/K. Returned tensors alias the graph pool."""
    def __init__(self, importance, similarity, k, author_select):
        validate(importance, similarity, k)
        if not callable(author_select):raise ValueError('missing author function')
        self.importance = torch.empty_like(importance)
        self.similarity = torch.empty_like(similarity)
        self.k = k
        self.importance.copy_(importance)
        self.similarity.copy_(similarity)
        torch.cuda.synchronize(importance.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.order, self.quotas = author_select(self.importance, self.similarity, k)

    def __call__(self, importance, similarity):
        validate(importance, similarity, self.k)
        if (importance.shape != self.importance.shape or similarity.shape != self.similarity.shape
            or importance.device != self.importance.device or similarity.device != self.similarity.device):
            raise ValueError('input differs from captured device or shape')
        self.importance.copy_(importance)
        self.similarity.copy_(similarity)
        self.graph.replay()
        return self.order, self.quotas
