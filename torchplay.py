import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def self_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Tokenize the input sentence
sentence = "I am Optimus prime, Therefore I'm on"
tokens = sentence.lower().split()

# Create embeddings (here we're using random embeddings for simplicity)
embed_dim = 64
embeddings = torch.randn(len(tokens), embed_dim)

# Compute self-attention
_, attention_weights = self_attention(embeddings, embeddings, embeddings)

# Visualize the attention pattern
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights.detach().numpy(), annot=True, cmap='YlGnBu', xticklabels=tokens, yticklabels=tokens)
plt.title("Attention Pattern")
plt.xlabel("Key/Value")
plt.ylabel("Query")
plt.show()

print("Attention weights shape:", attention_weights.shape)