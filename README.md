# Toiny-LM: A ChatML Language Model

A simple implementation of a transformer-based language model that uses OpenAI's ChatML format and can run on your MacBook's Metal GPU. This project demonstrates the core concepts behind modern conversational AI models in a simplified, educational format.

## Features

- **ChatML Format**: Uses OpenAI's ChatML format for structured conversations
- **Transformer Architecture**: Implements the core transformer components (multi-head attention, feed-forward networks, positional encoding)
- **Metal GPU Support**: Automatically uses your MacBook's Metal GPU for accelerated training and inference
- **Word-level Tokenization**: More sophisticated than character-level, better for conversational AI
- **Conversational AI**: Designed for question-answer interactions
- **Interactive Mode**: Chat with your trained model
- **Educational**: Clean, well-commented code perfect for learning about conversational LLMs

## Architecture

The model consists of:

1. **ChatMLTokenizer**: Word-level tokenizer that handles ChatML format with special tokens
2. **MultiHeadAttention**: Implements scaled dot-product attention with multiple heads
3. **FeedForward**: Position-wise feed-forward networks
4. **TransformerBlock**: Combines attention and feed-forward with residual connections
5. **ChatMLLM**: Main model with token embeddings, positional encoding, and transformer blocks

## Model Specifications

- **Vocabulary Size**: ~1000 words (configurable)
- **Model Dimensions**: 256 (configurable)
- **Attention Heads**: 8
- **Transformer Layers**: 6
- **Feed-forward Dimension**: 1024
- **Sequence Length**: 512 tokens
- **Parameters**: ~2.5M (larger than the character-level version for better conversational ability)

## ChatML Format

The model uses OpenAI's ChatML format for training data:

```
<|im_start|>system
You are a helpful AI assistant that provides informative and accurate responses.
<|im_end|>
<|im_start|>user
What is artificial intelligence?
<|im_end|>
<|im_start|>assistant
Artificial intelligence (AI) is a branch of computer science...
<|im_end|>
```

This format allows the model to learn:
- System instructions and context
- User questions and prompts
- Assistant responses
- Conversation structure

## Installation

The project uses `uv` for dependency management. Make sure you have the dependencies installed:

```bash
# Install dependencies (if not already done)
uv sync
```

## Usage

### 1. Training the Model

#### Using Hugging Face Datasets (Recommended)

```bash
# List available datasets
python main.py --list-datasets

# Train with a multi-turn conversation dataset (recommended)
python main.py --dataset "dim/norquinal_claude_multiround_chat_30k" --max-samples 5000

# Train with high-quality Puffin dataset
python main.py --dataset "LDJnr/Puffin"
```

#### Using Local File (Original)

```bash
python main.py
```

This will:
- Load training data (Hugging Face dataset if specified, otherwise `tiny_corpus.txt`)
- Automatically convert to ChatML format
- Create a word-level tokenizer with ChatML special tokens
- Train the transformer model for 15 epochs
- Generate sample responses
- Save the trained model to `output/chatml_llm.pth`

### 2. Running Inference

After training, you can use the model in two ways:

#### A. Generate Example Responses
```bash
python inference.py
```

#### B. Interactive Mode
```bash
python inference.py --interactive
```

In interactive mode, you can:
- Ask questions and get AI responses
- Adjust temperature with `temp <value>` (lower = more focused, higher = more creative)
- Adjust max length with `length <value>`
- Type `quit` or `exit` to stop

### 3. Example Commands

```bash
# List available Hugging Face datasets
python main.py --list-datasets

# Train with Hugging Face dataset
python main.py --dataset "LDJnr/Puffin"

# Train with limited samples from a large dataset
python main.py --dataset "dim/norquinal_claude_multiround_chat_30k" --max-samples 1000

# Train with local file (default)
python main.py

# Test dataset loading
python test_datasets.py

# Generate example responses
python inference.py

# Start interactive chat
python inference.py --interactive
```

## Training Data

The model supports multiple data sources:

### Hugging Face Datasets (NEW!)

You can now train on multi-turn conversation datasets from Hugging Face:

```bash
# List available datasets
python main.py --list-datasets

# Train with a specific dataset
python main.py --dataset "dim/norquinal_claude_multiround_chat_30k" --max-samples 1000

# Train with LDJnr/Puffin dataset
python main.py --dataset "LDJnr/Puffin"
```

**Supported Datasets:**
- `dim/norquinal_claude_multiround_chat_30k` - 30k multi-round conversations with Claude
- `LDJnr/Puffin` - High-quality multi-turn conversations covering various topics
- `argilla/synthetic-sft-customer-support-multi-turn` - Synthetic customer support conversations

### Local File (Original)

The model also comes with a local ChatML-formatted corpus in `tiny_corpus.txt` containing conversations about AI, machine learning, and technology. You can:

1. **Expand the corpus**: Add more ChatML conversations to `tiny_corpus.txt`
2. **Use your own data**: Replace the content with your own ChatML conversations
3. **Add more diverse topics**: Include different subjects and conversation styles

### Data Format

All datasets are automatically converted to ChatML format:
```
<|im_start|>system
You are a helpful AI assistant that provides informative and accurate responses.
<|im_end|>
<|im_start|>user
Your question here
<|im_end|>
<|im_start|>assistant
Assistant response here
<|im_end|>
```

## Customization

You can modify the model architecture in `main.py`:

```python
# In the main() function, adjust these parameters:
model = ChatMLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=256,        # Model dimension
    num_heads=8,        # Number of attention heads
    num_layers=6,       # Number of transformer layers
    d_ff=1024,         # Feed-forward dimension
    max_seq_len=512,   # Maximum sequence length
    dropout=0.1        # Dropout rate
)
```

## How It Works

1. **ChatML Parsing**: Text is parsed into structured conversations
2. **Tokenization**: Words are converted to token indices with special ChatML tokens
3. **Embedding**: Tokens are converted to dense vectors
4. **Positional Encoding**: Position information is added to embeddings
5. **Transformer Blocks**: Multiple layers of self-attention and feed-forward networks
6. **Output**: Final layer predicts the next token probabilities
7. **Generation**: Autoregressive sampling generates responses word by word

## Metal GPU Acceleration

The model automatically detects and uses your MacBook's Metal GPU:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

This provides significant speedup compared to CPU-only training.

## Advantages of ChatML Format

- **Structured Conversations**: Clear separation between system, user, and assistant messages
- **Better Training**: Model learns conversation patterns and roles
- **More Realistic**: Mimics real chatbot interactions
- **Extensible**: Easy to add new conversation types or roles

## Limitations

This is a simplified educational implementation:

- **Small Model**: Only ~2.5M parameters vs billions in commercial LLMs
- **Limited Training Data**: Small corpus for demonstration
- **Word-level**: Uses words instead of subword tokens like BPE
- **No Advanced Features**: No techniques like attention optimization, model parallelism, etc.

## Learning Resources

To understand the concepts better:

1. **ChatML Format**: OpenAI's documentation on ChatML
2. **Attention Is All You Need**: The original transformer paper
3. **The Illustrated Transformer**: Visual explanation of transformers
4. **Conversational AI**: Research on dialogue systems and chatbots

## Future Enhancements

Potential improvements you could implement:

- Subword tokenization (BPE, WordPiece)
- Larger model size and training data
- Advanced sampling techniques (top-k, nucleus sampling)
- Attention optimizations
- Model quantization for faster inference
- Multi-turn conversation support
- Context window management

## License

This project is for educational purposes. Feel free to modify and experiment!
