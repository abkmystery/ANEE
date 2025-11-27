import torch
import torch.nn.functional as F


class ANEEProfiler:
    """
    Extracts simple, interpretable signals from hidden states and logits.
    These will form the state vector for the learned controller.
    """

    # @staticmethod
    # def compute_entropy_and_max_prob(logits: torch.Tensor) -> tuple[float, float]:
    #     """
    #     Entropy and max probability of the next-token distribution.
    #     Assumes logits shape: (batch, seq_len, vocab_size); batch=1.
    #     Uses the last token position.
    #     """
    #     probs = F.softmax(logits[:, -1, :], dim=-1)
    #     log_probs = torch.log(probs + 1e-12)
    #     entropy = -(probs * log_probs).sum(dim=-1)  # (1,)
    #     max_prob, _ = probs.max(dim=-1)
    #     return entropy.item(), max_prob.item()

    @staticmethod
    def hidden_norm(h: torch.Tensor) -> float:
        """
        L2 norm of hidden state at the last token.
        h: (batch, seq_len, hidden_dim)
        """
        return torch.norm(h[:, -1, :], p=2).item()

    @staticmethod
    def delta_hidden_norm(h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
        """
        ||h_curr - h_prev||_2 at the last token.
        """
        diff = h_curr[:, -1, :] - h_prev[:, -1, :]
        return torch.norm(diff, p=2).item()

    @staticmethod
    def variance(h: torch.Tensor) -> float:
        """
        Variance over hidden dims at the last token.
        """
        return torch.var(h[:, -1, :]).item()
