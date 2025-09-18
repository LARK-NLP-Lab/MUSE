import torch
import torch.nn.functional as F

def compute_answer_likelihood(model, tokenizer, input_text):
    """
    Computes the log likelihood of the sequence ending with 'Answer is: Yes' and 'Answer is: No'.
    
    Args:
        model: The language model (e.g., a Hugging Face transformer model).
        tokenizer: The associated tokenizer.
        input_text: The input text to be conditioned on.
    
    Returns:
        A dictionary with log probabilities of "Answer is: Yes" and "Answer is: No".
    """
    # Define possible completions
    completion_yes = " Answer is: Yes."
    completion_no = " Answer is: No."
    
    # Tokenize inputs
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_len = input_ids.shape[-1]  # Length of the input prompt
    
    # Encode full sequences
    yes_ids = tokenizer.encode(input_text + completion_yes, return_tensors="pt")
    no_ids = tokenizer.encode(input_text + completion_no, return_tensors="pt")

    # Get model outputs
    with torch.no_grad():
        yes_outputs = model(yes_ids)
        no_outputs = model(no_ids)

    # Extract log probabilities for generated part
    yes_logits = yes_outputs.logits[:, input_len-1:-1, :]  # Extract logits for new tokens
    no_logits = no_outputs.logits[:, input_len-1:-1, :]

    # Compute probabilities of each token in the respective completions
    yes_probs = F.log_softmax(yes_logits, dim=-1)
    no_probs = F.log_softmax(no_logits, dim=-1)

    # Extract token IDs for "Yes" and "No" completions
    yes_token_ids = yes_ids[:, input_len:]  # Get only the new tokens
    no_token_ids = no_ids[:, input_len:]

    # Compute sequence log likelihood by summing token log probabilities
    yes_log_prob = sum(yes_probs[0, i, token_id].item() for i, token_id in enumerate(yes_token_ids[0]))
    no_log_prob = sum(no_probs[0, i, token_id].item() for i, token_id in enumerate(no_token_ids[0]))

    return {
        "prob_yes": torch.exp(torch.tensor(yes_log_prob)).item(),
        "prob_no": torch.exp(torch.tensor(no_log_prob)).item()
    }