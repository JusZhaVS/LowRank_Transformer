import torch
import tqdm


def calculate_perplexity(model, tokenizer, test_prompts, batch_size: int = 16, max_length: int = 512):
    """
    Calculate perplexity of model with test prompts in a batched way.
    Applies attention mask so that only valid tokens contribute to the loss.
    """
    model.eval()
    device = next(model.parameters()).device

    print("Calculating perplexity of model with test prompts (batched):")
    print("=" * 60)

    # Tokenize all prompts at once (batched)
    encodings = tokenizer(
        test_prompts,
        max_length=max_length,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    num_samples = input_ids.size(0)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size)):
            end = min(start + batch_size, num_samples)
            batch_input_ids = input_ids[start:end]

            # Forward pass
            batch_attention_mask = attention_mask[start:end] if attention_mask is not None else None
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits # Explicitly access logits attribute

            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_input_ids[..., 1:].contiguous()
            if attention_mask is not None:
                batch_attention_mask = attention_mask[start:end]
                shift_mask = batch_attention_mask[..., 1:].contiguous()
            else:
                shift_mask = None

            # Flatten for loss computation
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )  # (batch * seq_len-1,)

            if shift_mask is not None:
                loss = loss * shift_mask.view(-1).float()
                num_valid = shift_mask.sum().item()
            else:
                num_valid = shift_labels.numel()

            total_loss += loss.sum().item()
            total_tokens += num_valid

    avg_loss = total_loss / max(1, total_tokens)
    avg_perplexity = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    return avg_perplexity

print("calculate_perplexity function redefined with explicit logits access.")
