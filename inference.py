import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import List
import sys
import os

# Import the model classes from main.py
from main import ChatMLLM, ChatMLTokenizer

# Check if Metal GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path: str = 'output/chatml_llm.pth'):
    """Load the trained ChatML model and tokenizer"""
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate tokenizer
    vocab = checkpoint['tokenizer_vocab']
    word_to_idx = checkpoint['word_to_idx']
    tokenizer = ChatMLTokenizer.__new__(ChatMLTokenizer)
    tokenizer.vocab = vocab
    tokenizer.word_to_idx = word_to_idx
    tokenizer.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    tokenizer.vocab_size = len(vocab)
    
    # Recreate model
    config = checkpoint['model_config']
    model = ChatMLLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_chat_response_interactive(model: ChatMLLM, tokenizer: ChatMLTokenizer, 
                                     user_input: str, max_length: int = 100, temperature: float = 1.0):
    """Generate ChatML response using the trained model"""
    
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

def interactive_mode(model: ChatMLLM, tokenizer: ChatMLTokenizer):
    """Run the ChatML model in interactive mode"""
    
    print("\n" + "="*60)
    print("ChatML LLM Interactive Mode")
    print("="*60)
    print("Ask questions and the AI will respond in ChatML format.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'temp <value>' to change temperature (default: 1.0)")
    print("  - Type 'length <value>' to change max length (default: 100)")
    print("="*60)
    
    temperature = 1.0
    max_length = 100
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to: {temperature}")
                    continue
                except (IndexError, ValueError):
                    print("Invalid temperature value. Use format: temp <number>")
                    continue
            
            elif user_input.lower().startswith('length '):
                try:
                    max_length = int(user_input.split()[1])
                    print(f"Max length set to: {max_length}")
                    continue
                except (IndexError, ValueError):
                    print("Invalid length value. Use format: length <number>")
                    continue
            
            elif not user_input:
                continue
            
            # Generate response
            print(f"\nGenerating with temperature={temperature}, max_length={max_length}...")
            response = generate_chat_response_interactive(model, tokenizer, user_input, max_length, temperature)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function for ChatML inference"""
    
    # Load the model
    print("Loading ChatML model...")
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return
    
    print("ChatML model loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode(model, tokenizer)
    else:
        # Generate some example responses
        print("\nGenerating example responses...")
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "What is deep learning?",
            "How do transformers work?"
        ]
        
        for question in test_questions:
            response = generate_chat_response_interactive(model, tokenizer, question, max_length=80, temperature=0.8)
            print(f"\nQ: {question}")
            print(f"A: {response}")
            print("-" * 60)

if __name__ == "__main__":
    main() 