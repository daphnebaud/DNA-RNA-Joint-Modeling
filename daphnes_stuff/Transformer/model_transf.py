import torch
import torch.nn as nn
import math
import copy

# The multi-head attention mechanism computes the attention between each pair of positions in a sequence.
# It consists of multiple “attention heads” that capture different aspects of the input sequence.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            # For causal masking (like in decoders), mask == 0 would be -inf
            # For encoder padding mask, mask == 0 indicates padding.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs # Also return attention probabilities if needed for visualization/interpretability

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs # Return output and attention probabilities


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Positional Encoding is used to inject the position information of each token in the input sequence.
# It uses sine and cosine functions of different frequencies to generate the positional encoding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# The EncoderLayer class defines a single layer of the transformer's encoder
# multi-head self-attention mechanism -> position-wise feed-forward neural network, with residual connections, layer normalization, and dropout applied as appropriate
# Together, these components allow the encoder to capture complex relationships in the input data and transform them into a useful representation for downstream tasks
# Typically, multiple such encoder layers are stacked to form the complete encoder part of a transformer model.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, attn_probs = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_probs # Return attention probabilities for potential analysis


# The DNASequenceClassifier (a modified Transformer Encoder) for binary classification
class DNASequenceClassifier(nn.Module):
    def __init__(self, input_features, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        """
        Initializes the DNASequenceClassifier.

        Args:
            input_features (int): Number of input features per position (e.g., 5 for one-hot A,C,G,T,N).
            d_model (int): Dimension of the model's embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            d_ff (int): Dimension of the feed-forward network.
            max_seq_length (int): Maximum sequence length for positional encoding.
            dropout (float): Dropout rate.
        """
        super(DNASequenceClassifier, self).__init__()

        # Input projection from one-hot features to d_model dimension
        # Your one-hot encoded sequence is (batch_size, seq_len, num_nucleotides=5)
        # We need to project num_nucleotides to d_model
        self.input_projection = nn.Linear(input_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Classification head for binary output (expressed/not expressed)
        # We'll take the representation of the entire sequence (e.g., by global average pooling)
        # and pass it through a linear layer to get a single output score.
        self.classification_head = nn.Linear(d_model, 1)


    def forward(self, src):
        """
        Forward pass for the DNA Sequence Classifier.

        Args:
            src (torch.Tensor): Input sequence tensor.
                                Expected shape: (batch_size, sequence_length, input_features)
                                (e.g., batch_size, window_size, 5 for one-hot)
        Returns:
            torch.Tensor: Logits for binary classification (shape: batch_size, 1).
        """
        # Create a padding mask if your sequences can have variable lengths
        # For fixed window_size, this mask might not be strictly necessary if all inputs are full.
        # If padding will be used, adapt this:
        # src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2) # Assumes 0 sum for padded one-hot vectors
        # For now, let's assume fixed length or no padding mask needed
        src_mask = None # For simplicity, assuming no padding or constant window size

        # Project input features to d_model and add positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.input_projection(src)))

        enc_output = src_embedded
        # Store attention probabilities if you want to analyze them
        all_attn_probs = []
        for enc_layer in self.encoder_layers:
            enc_output, attn_probs = enc_layer(enc_output, src_mask)
            all_attn_probs.append(attn_probs)

        # --- Aggregation for Classification ---
        # Option 1: Global Average Pooling (common for sequence classification)
        # Averages the representations across the sequence length dimension
        pooled_output = enc_output.mean(dim=1) # Shape: (batch_size, d_model)

        # Option 2: Use the representation of the first token (if you add a [CLS] token like in BERT)
        # pooled_output = enc_output[:, 0, :]

        # Option 3: Max Pooling
        # pooled_output = enc_output.max(dim=1)[0]


        # Pass the pooled representation through the classification head
        logits = self.classification_head(pooled_output) # Shape: (batch_size, 1)

        return logits # We return logits (raw scores) for BCEWithLogitsLoss