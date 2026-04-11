import torch
import torch.nn as nn


def test_weight_tying(word_embedding, lm_head, vocab_size=10, test_tokens=None):
    """Test if weight tying is working correctly."""

    print("=== Weight Tying Test ===")

    # Check if weights are actually tied
    print(f"Weights are tied: {lm_head.weight is word_embedding.weight}")
    print(f"Weight shapes - Embedding: {word_embedding.weight.shape}, LM Head: {lm_head.weight.shape}")

    # Test with a few token IDs
    if test_tokens is None:
        test_tokens = torch.tensor([0, 1, 2, 100, 500])  # Adjust based on your vocab size

    print(f"\nTesting with tokens: {test_tokens.tolist()}")

    # Forward pass: tokens -> embeddings -> logits
    embeddings = word_embedding(test_tokens)  # [num_tokens, embed_dim]
    logits = lm_head(embeddings)  # [num_tokens, vocab_size]

    # Get predicted tokens (highest logit)
    predicted_tokens = torch.argmax(logits, dim=-1)

    print(f"Original tokens:   {test_tokens.tolist()}")
    print(f"Predicted tokens:  {predicted_tokens.tolist()}")
    print(f"Match rate: {(test_tokens == predicted_tokens).float().mean().item():.2%}")

    # Check logit values for original tokens
    print("\nLogit analysis:")
    for i, token_id in enumerate(test_tokens):
        original_logit = logits[i, token_id].item()
        max_logit = logits[i].max().item()
        max_token = logits[i].argmax().item()
        print(f"Token {token_id}: logit={original_logit:.3f}, max_logit={max_logit:.3f} (token {max_token})")

    # Test with bias removed (if bias exists)
    if lm_head.bias is not None:
        print("\n=== Test without bias ===")
        logits_no_bias = lm_head(embeddings) - lm_head.bias
        predicted_no_bias = torch.argmax(logits_no_bias, dim=-1)
        print(f"Predicted (no bias): {predicted_no_bias.tolist()}")
        print(f"Match rate (no bias): {(test_tokens == predicted_no_bias).float().mean().item():.2%}")

    #compute ce loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, test_tokens)
    print(f"\nCross-Entropy Loss: {loss.item():.4f}")

# Example usage with your factory
def test_your_model():
    """Test your specific model setup."""

    # Create your model using the factory
    # factory = TransformerNetModelFactory(your_config)
    # modules = factory._create_modules()
    # word_embedding = modules.word_embedding
    # lm_head = modules.lm_head

    # For demonstration - replace with your actual modules
    vocab_size = 30522  # BERT vocab size
    embed_dim = 768     # BERT hidden size

    # Create dummy embeddings to simulate your setup
    word_embedding = nn.Embedding(vocab_size, embed_dim)
    lm_head = nn.Linear(embed_dim, vocab_size, bias=True)

    # Simulate tied weights
    lm_head.weight = word_embedding.weight

    # Test
    test_weight_tying(word_embedding, lm_head, vocab_size)

# Run the test
if __name__ == "__main__":
    test_your_model()
