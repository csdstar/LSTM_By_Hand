import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self, embedding_weight: torch.Tensor, num_sampled: int):
        super(NCELoss, self).__init__()
        self.register_buffer('embedding_weight', embedding_weight)  # shape: [vocab_size, embed_dim]
        self.num_sampled = num_sampled
        self.vocab_size = embedding_weight.shape[0]
        self.embed_dim = embedding_weight.shape[1]

    def forward(self, y_pred: torch.Tensor, y_true_indices: torch.Tensor):
        """
        y_pred: [batch, embed_dim]     - 模型预测输出
        y_true_indices: [batch]        - 标签词索引
        """
        batch_size = y_pred.shape[0]
        device = y_pred.device

        # 正样本嵌入
        true_embed = self.embedding_weight[y_true_indices]  # [batch, embed_dim]
        true_logits = (y_pred * true_embed).sum(dim=-1)  # [batch]

        # 负样本索引：[batch, num_sampled]
        neg_indices = torch.randint(0, self.vocab_size, (batch_size, self.num_sampled), device=device)
        neg_embed = self.embedding_weight[neg_indices]  # [batch, num_sampled, embed_dim]

        # 扩展 y_pred → [batch, 1, embed_dim]
        y_pred_expanded = y_pred.unsqueeze(1)

        # 计算负样本得分
        neg_logits = torch.bmm(neg_embed, y_pred_expanded.transpose(1, 2)).squeeze(-1)  # [batch, num_sampled]

        # NCE Loss = -log(sigmoid(true)) - sum log(sigmoid(neg)^-1)
        loss_pos = F.logsigmoid(true_logits)             # [batch]
        loss_neg = F.logsigmoid(-neg_logits).sum(dim=1)  # [batch]
        loss = -(loss_pos + loss_neg).mean()

        # 计算 dL/dY_pred
        with torch.no_grad():
            # 正样本梯度
            pos_grad = torch.sigmoid(-true_logits).unsqueeze(1) * true_embed  # [batch, embed_dim]

            # 负样本梯度
            neg_sigmoid = torch.sigmoid(neg_logits)  # [batch, num_sampled]
            neg_grad = (neg_sigmoid.unsqueeze(2) * neg_embed).sum(dim=1)  # [batch, embed_dim]

            dY = (pos_grad - neg_grad) / batch_size  # 归一化梯度

        return loss.detach(), dY
