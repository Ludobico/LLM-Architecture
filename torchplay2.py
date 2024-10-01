import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def self_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Tokenize the input sentence and add keywords
sentence = "I am Optimus prime, Therefore I'm on"
tokens = sentence.lower().split() + ["autobot", "decepticon"]

# Create embeddings (using random embeddings for simplicity)
embed_dim = 64
embeddings = torch.randn(len(tokens), embed_dim)

# Compute self-attention
_, attention_weights = self_attention(embeddings, embeddings, embeddings)

# Visualize the attention pattern
plt.figure(figsize=(12, 10))
sns.heatmap(attention_weights.detach().numpy(), annot=True, cmap='YlGnBu', xticklabels=tokens, yticklabels=tokens)
plt.title("Attention Pattern (Including Autobot and Decepticon)")
plt.xlabel("Key/Value")
plt.ylabel("Query")
plt.show()

# Analyze attention to "Optimus prime"
optimus_idx = tokens.index("prime")
autobot_idx = tokens.index("autobot")
decepticon_idx = tokens.index("decepticon")

optimus_attention = attention_weights[:, optimus_idx].detach().numpy()
autobot_attention = attention_weights[:, autobot_idx].detach().numpy()
decepticon_attention = attention_weights[:, decepticon_idx].detach().numpy()

# Compare attentions
plt.figure(figsize=(10, 6))
plt.bar(tokens, optimus_attention, alpha=0.5, label='Attention to "prime"')
plt.bar(tokens, autobot_attention, alpha=0.5, label='Attention to "autobot"')
plt.bar(tokens, decepticon_attention, alpha=0.5, label='Attention to "decepticon"')
plt.title("Attention Comparison")
plt.xlabel("Tokens")
plt.ylabel("Attention Score")
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate similarity scores
optimus_embed = embeddings[optimus_idx]
autobot_embed = embeddings[autobot_idx]
decepticon_embed = embeddings[decepticon_idx]

autobot_score = F.cosine_similarity(optimus_embed, autobot_embed, dim=0)
decepticon_score = F.cosine_similarity(optimus_embed, decepticon_embed, dim=0)

print(f"Similarity score with Autobot: {autobot_score.item():.4f}")
print(f"Similarity score with Decepticon: {decepticon_score.item():.4f}")

if autobot_score > decepticon_score:
    print("The context seems more related to Autobots.")
else:
    print("The context seems more related to Decepticons.")