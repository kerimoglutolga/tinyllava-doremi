from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class CausalLMOutputWithPerTokenLoss(CausalLMOutputWithPast):
    per_token_loss: torch.Tensor = None
    token_mask: torch.Tensor = None

@dataclass
class CausalLMOutputForDoReMi(CausalLMOutputWithPerTokenLoss):
    _domains: torch.LongTensor = None # (batch_size, 1)
    reference_per_token_loss: torch.Tensor = None

    @cached_property
    def domains(self):
        batch_size = self._domains.size(0) 
        expanded_domains = self._domains.expand(batch_size, self.logits.size(1) - 1)
        flattened_domains = expanded_domains.flatten()
        return flattened_domains
    
    @cached_property
    def per_domain_losses(self):
        flat_losses = self.per_token_loss - self.reference_per_token_loss # shape: (N,)
        flat_mask = self.token_mask.bool() # shape: (N,)
        flat_domains = self.domains # shape: (N,)

        text_losses = self.per_token_loss[flat_domains == 3].sum()
        reference_text_losses = self.reference_per_token_loss[flat_domains == 3].sum()
        logger.info(f"Num text samples: {flat_domains[flat_domains == 3].size(0)}")
        logger.info(f"Text Losses: {text_losses}")
        logger.info(f"Reference Text Losses: {reference_text_losses}")
        
        # Select only the tokens that are valid (according to token_mask)
        valid_losses = torch.clip(flat_losses[flat_mask], min=0)

        valid_domains = flat_domains[flat_mask]
        
        num_domains = 6 # Don't hardcode this ideally
        
        # Initialize accumulators for loss sum and token counts per domain.
        domain_loss_sum = torch.zeros(
            num_domains, dtype=valid_losses.dtype, device=valid_losses.device
        )
        domain_token_count = torch.zeros(
            num_domains, dtype=valid_losses.dtype, device=valid_losses.device
        )
        
        # Scatter-add the valid losses into the sum accumulator based on domain indices.
        domain_loss_sum.scatter_add_(0, valid_domains, valid_losses)
        # Similarly, accumulate a count for each valid token.
        ones = torch.ones_like(valid_losses)
        domain_token_count.scatter_add_(0, valid_domains, ones)
        
        # Use all_reduce to sum up the accumulators across all devices.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(domain_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(domain_token_count, op=torch.distributed.ReduceOp.SUM)

        # Compute the per-domain mean loss, avoiding division by zero.
        per_domain_loss = domain_loss_sum / domain_token_count.clamp(min=1)

        logger.info(f"Per-domain loss: {per_domain_loss}")
        
        return per_domain_loss
        
