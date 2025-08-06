import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
from typing import List, Tuple, Dict
import json
import os
from data_loader import DataLoader as HFDataLoader, print_available_datasets

# Check if Metal GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ChatMLTokenizer:
    """Tokenizer for ChatML format conversations"""
    
    def __init__(self, text: str, vocab_size: int = 1000):
        # Special tokens for ChatML format
        self.special_tokens = {
            '<|im_start|>': 0,
            '<|im_end|>': 1,
            '<|im_start|>system': 2,
            '<|im_start|>user': 3,
            '<|im_start|>assistant': 4,
            '<|pad>': 5,
            '<|unk>': 6
        }
        
        # Extract all text content (excluding ChatML tags)
        content_text = self.extract_content(text)
        
        # Create vocabulary from most common words
        words = self.tokenize_text(content_text)
        word_counts = Counter(words)
        most_common_words = [word for word, _ in word_counts.most_common(vocab_size - len(self.special_tokens))]
        
        # Build vocabulary
        self.vocab = list(self.special_tokens.keys()) + most_common_words
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def extract_content(self, text: str) -> str:
        """Extract actual content from ChatML format"""
        # Remove ChatML tags and keep only the content
        content = re.sub(r'<\|im_start\|>.*?\n', '', text)
        content = re.sub(r'<\|im_end\|>\n', '', text)
        return content
    
    def tokenize_text(self, text: str) -> List[str]:
        """Simple word-level tokenization"""
        # Split on whitespace and basic punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token indices"""
        tokens = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('<|im_start|>'):
                if line == '<|im_start|>':
                    tokens.append(self.word_to_idx['<|im_start|>'])
                elif line == '<|im_start|>system':
                    tokens.append(self.word_to_idx['<|im_start|>system'])
                elif line == '<|im_start|>user':
                    tokens.append(self.word_to_idx['<|im_start|>user'])
                elif line == '<|im_start|>assistant':
                    tokens.append(self.word_to_idx['<|im_start|>assistant'])
            elif line == '<|im_end|>':
                tokens.append(self.word_to_idx['<|im_end|>'])
            else:
                # Tokenize the content
                words = self.tokenize_text(line)
                for word in words:
                    tokens.append(self.word_to_idx.get(word, self.word_to_idx['<|unk>']))
        
        return tokens
    
    def decode(self, indices: List[int]) -> str:
        """Decode token indices back to text"""
        words = []
        for idx in indices:
            if idx < len(self.idx_to_word):
                words.append(self.idx_to_word[idx])
            else:
                words.append('<|unk>')
        return ' '.join(words)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class ChatMLLM(nn.Module):
    """ChatML Language Model using Transformer architecture"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, max_seq_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_len: int, d_model: int):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def create_causal_mask(self, seq_len: int):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # Token embeddings
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        # Apply transformer blocks
        causal_mask = self.create_causal_mask(seq_len) if mask is None else mask
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Output layer
        logits = self.output_layer(x)
        return logits

class ChatMLDataset(Dataset):
    """Dataset for ChatML conversation data"""
    
    def __init__(self, text: str, tokenizer: ChatMLTokenizer, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(0, len(tokens) - seq_len):
            sequence = tokens[i:i + seq_len]
            target = tokens[i + 1:i + seq_len + 1]
            self.sequences.append(sequence)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.long),
                torch.tensor(self.targets[idx], dtype=torch.long))

def train_model(model: ChatMLLM, dataloader: DataLoader, 
                num_epochs: int = 10, learning_rate: float = 0.0001):
    """Train the ChatML language model"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return model

def generate_chat_response(model: ChatMLLM, tokenizer: ChatMLTokenizer, 
                          user_input: str, max_length: int = 100, temperature: float = 1.0):
    """Generate ChatML response using the trained model"""
    
    model.eval()
    model = model.to(device)
    
    # Format input as ChatML
    chatml_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize the input
    tokens = tokenizer.encode(chatml_input)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Update input sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Keep only the last max_seq_len tokens
            if tokens.size(1) > model.max_seq_len:
                tokens = tokens[:, -model.max_seq_len:]
            
            # Stop if we generate the end token
            if next_token.item() == tokenizer.word_to_idx['<|im_end|>']:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

def main(dataset_name: str = None, max_samples: int = None):
    """Main function to train and test the ChatML LLM
    
    Args:
        dataset_name: Hugging Face dataset name (e.g., "dim/norquinal_claude_multiround_chat_30k")
        max_samples: Maximum number of samples to use from dataset
    """
    
    # Initialize data loader
    data_loader = HFDataLoader()
    
    # Load training data - try Hugging Face dataset first, then fallback to local file
    text = None
    
    if dataset_name:
        try:
            print(f"Attempting to load Hugging Face dataset: {dataset_name}")
            text = data_loader.load_data(dataset_name=dataset_name, max_samples=max_samples)
            print(f"Successfully loaded data from {dataset_name}")
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            print("Falling back to local file...")
    
    # Fallback to local file if dataset loading failed or wasn't specified
    if text is None:
        try:
            text = data_loader.load_data(local_file='tiny_corpus.txt')
            print("Successfully loaded data from local file: tiny_corpus.txt")
        except Exception as e:
            print(f"Failed to load local file: {e}")
            print("\nAvailable Hugging Face datasets:")
            print_available_datasets()
            raise ValueError("Could not load any training data. Please check your dataset name or ensure tiny_corpus.txt exists.")
    
    print(f"Training text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = ChatMLTokenizer(text, vocab_size=1000)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    dataset = ChatMLDataset(text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Number of training sequences: {len(dataset)}")
    
    # Create model
    model = ChatMLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print("\nStarting training...")
    model = train_model(model, dataloader, num_epochs=15, learning_rate=0.0001)
    
    # Generate some responses
    print("\nGenerating responses...")
    test_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain artificial intelligence",
        "What is deep learning?",
        "How does natural language processing work?"
    ]
    
    for question in test_questions:
        response = generate_chat_response(model, tokenizer, question, max_length=50, temperature=0.8)
        print(f"\nQ: {question}")
        print(f"A: {response}")
        print("-" * 60)
    
    # Save the model
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'word_to_idx': tokenizer.word_to_idx,
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'd_model': model.d_model,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'max_seq_len': model.max_seq_len
        }
    }, 'output/chatml_llm.pth')
    
    print("\nModel saved to 'output/chatml_llm.pth'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ChatML Language Model")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Hugging Face dataset name (e.g., 'dim/norquinal_claude_multiround_chat_30k')")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to use from dataset")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available Hugging Face datasets")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print_available_datasets()
    else:
        main(dataset_name=args.dataset, max_samples=args.max_samples)
