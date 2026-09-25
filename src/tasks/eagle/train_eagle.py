import torch
from torch.amp import autocast  # type: ignore[attr-defined]


def compute_eagle_loss(
    eagle_module,
    base_model,
    batch,
    device=None,
):
    if device is None:
        device = next(base_model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_logprobs = batch["topk_logprobs"].to(device)
    target_indices = batch["topk_logprobs"].to(device)
    label_mask = batch["label_mask"].to(device)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        with torch.no_grad()
            base_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            base_hidden_states = base_outputs.hidden_states[-1]

        pred_hidden, pred_logits = eagle_module(
            token_id=input_ids,
            hidden_state=base_hidden_states[:, :-1, :]
        )

        shifted_target_logprods = target_logprobs[:, 1:, :]
        matched_target_indices = target_indices[:, 1:, :]
        mask = label_mask[:, 1:, :]

        gathered_student_logprods = shifted_target_logprods.gather(
            dim=-1,
            index=matched_target_indices
        )

        cross_entropy_loss = -(torch.exp(matched_target_indices)*gathered_student_logprods).sum(-1)

        final_loss = (cross_entropy_loss*mask).sum() / mask.sum().clamp(min=1)

    return final_loss
