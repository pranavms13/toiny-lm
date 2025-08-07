"""
Data loader module for handling different data sources including Hugging Face datasets
"""
import os
from typing import Optional, List, Dict, Any
from datasets import load_dataset, Dataset
import json


class DataLoader:
    """Flexible data loader that can handle local files and Hugging Face datasets"""
    
    def __init__(self):
        self.supported_hf_datasets = {
            # Multi-turn conversation datasets available on Hugging Face
            "dim/norquinal_claude_multiround_chat_30k": "conversations",
            "LDJnr/Puffin": "conversations", 
            "argilla/synthetic-sft-customer-support-multi-turn": "messages",
            "Salesforce/dialogstudio": "log",  # Uses 'log' field for conversations
        }
    
    def load_data(self, 
                  dataset_name: Optional[str] = None, 
                  local_file: Optional[str] = None,
                  split: str = "train",
                  max_samples: Optional[int] = None) -> str:
        """
        Load data from various sources and return ChatML formatted text
        
        Args:
            dataset_name: Hugging Face dataset name (e.g., "dim/norquinal_claude_multiround_chat_30k")
            local_file: Path to local file
            split: Dataset split to use (train, test, validation)
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            ChatML formatted text string
        """
        if dataset_name:
            print(f"Loading dataset from Hugging Face: {dataset_name}")
            return self._load_from_huggingface(dataset_name, split, max_samples)
        elif local_file and os.path.exists(local_file):
            print(f"Loading data from local file: {local_file}")
            return self._load_from_local_file(local_file)
        else:
            raise ValueError("Either dataset_name or valid local_file must be provided")
    
    def _load_from_local_file(self, file_path: str) -> str:
        """Load data from local file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_from_huggingface(self, dataset_name: str, split: str, max_samples: Optional[int]) -> str:
        """Load data from Hugging Face dataset and convert to ChatML format"""
        try:
            # Handle special cases for specific datasets
            if dataset_name == "Salesforce/dialogstudio":
                # DialogStudio has multiple subsets, let's use a popular one
                dataset = load_dataset(dataset_name, "MULTIWOZ2_2", split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                
            print(f"Loaded {len(dataset)} samples from {dataset_name}")
            
            # Convert to ChatML format based on dataset structure
            return self._convert_to_chatml(dataset, dataset_name)
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            print("Available datasets:")
            for name in self.supported_hf_datasets.keys():
                print(f"  - {name}")
            raise
    
    def _convert_to_chatml(self, dataset: Dataset, dataset_name: str) -> str:
        """Convert dataset to ChatML format"""
        chatml_conversations = []
        
        for sample in dataset:
            conversation = self._extract_conversation(sample, dataset_name)
            if conversation:
                chatml_conversations.append(conversation)
        
        return "\n".join(chatml_conversations)
    
    def _extract_conversation(self, sample: Dict[str, Any], dataset_name: str) -> Optional[str]:
        """Extract conversation from a dataset sample and format as ChatML"""
        try:
            if dataset_name == "dim/norquinal_claude_multiround_chat_30k":
                return self._convert_claude_format(sample.get("conversations", []))
            
            elif dataset_name == "LDJnr/Puffin":
                return self._convert_puffin_format(sample.get("conversations", []))
            
            elif dataset_name == "argilla/synthetic-sft-customer-support-multi-turn":
                return self._convert_argilla_format(sample.get("messages", []))
            
            elif dataset_name == "Salesforce/dialogstudio":
                return self._convert_dialogstudio_format(sample.get("log", []))
            
            else:
                # Try to auto-detect format
                return self._auto_detect_format(sample)
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def _convert_claude_format(self, conversations: List[Dict]) -> str:
        """Convert Claude multiround chat format to ChatML"""
        if not conversations:
            return ""
        
        chatml_parts = [
            "<|im_start|>system",
            "You are a helpful AI assistant that provides informative and accurate responses.",
            "<|im_end|>"
        ]
        
        for msg in conversations:
            role = msg.get("from", "").lower()
            content = msg.get("value", "").strip()
            
            if role == "human":
                chatml_parts.extend([
                    "<|im_start|>user",
                    content,
                    "<|im_end|>"
                ])
            elif role in ["gpt", "assistant"]:
                chatml_parts.extend([
                    "<|im_start|>assistant", 
                    content,
                    "<|im_end|>"
                ])
        
        return "\n".join(chatml_parts)
    
    def _convert_puffin_format(self, conversations: List[Dict]) -> str:
        """Convert Puffin format to ChatML"""
        return self._convert_claude_format(conversations)  # Same format as Claude
    
    def _convert_argilla_format(self, messages: List[Dict]) -> str:
        """Convert Argilla customer support format to ChatML"""
        if not messages:
            return ""
        
        chatml_parts = [
            "<|im_start|>system",
            "You are a helpful customer support assistant.",
            "<|im_end|>"
        ]
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()
            
            if role == "user":
                chatml_parts.extend([
                    "<|im_start|>user",
                    content,
                    "<|im_end|>"
                ])
            elif role in ["assistant", "agent"]:
                chatml_parts.extend([
                    "<|im_start|>assistant",
                    content, 
                    "<|im_end|>"
                ])
        
        return "\n".join(chatml_parts)
    
    def _convert_dialogstudio_format(self, log: List[Dict]) -> str:
        """Convert DialogStudio format to ChatML"""
        if not log:
            return ""
        
        chatml_parts = [
            "<|im_start|>system",
            "You are a helpful AI assistant that provides informative and accurate responses.",
            "<|im_end|>"
        ]
        
        for turn in log:
            user_utterance = turn.get("user utterance", "").strip()
            system_response = turn.get("system response", "").strip()
            
            if user_utterance:
                chatml_parts.extend([
                    "<|im_start|>user",
                    user_utterance,
                    "<|im_end|>"
                ])
            
            if system_response:
                chatml_parts.extend([
                    "<|im_start|>assistant",
                    system_response,
                    "<|im_end|>"
                ])
        
        return "\n".join(chatml_parts)
    
    def _auto_detect_format(self, sample: Dict[str, Any]) -> Optional[str]:
        """Try to auto-detect conversation format"""
        # Look for common conversation fields
        possible_fields = ["conversations", "messages", "dialogue", "turns", "log"]
        
        for field in possible_fields:
            if field in sample and isinstance(sample[field], list):
                # Try to detect the format based on content
                conv_data = sample[field]
                if conv_data and isinstance(conv_data[0], dict):
                    first_msg = conv_data[0]
                    
                    # Check for Claude/Puffin style format
                    if "from" in first_msg and "value" in first_msg:
                        return self._convert_claude_format(conv_data)
                    
                    # Check for standard role/content format
                    elif "role" in first_msg and "content" in first_msg:
                        return self._convert_argilla_format(conv_data)
        
        return None


def get_alternative_datasets():
    """Return list of alternative multi-turn datasets if SoftAge-AI dataset is not available"""
    return [
        {
            "name": "dim/norquinal_claude_multiround_chat_30k",
            "description": "30k multi-round conversations with Claude",
            "size": "32.2k rows",
            "format": "conversations list with from/value fields"
        },
        {
            "name": "LDJnr/Puffin", 
            "description": "High-quality multi-turn conversations covering various topics",
            "size": "3k rows",
            "format": "conversations list with from/value fields"
        },
        {
            "name": "argilla/synthetic-sft-customer-support-multi-turn",
            "description": "Synthetic customer support multi-turn conversations",
            "size": "100 rows",
            "format": "messages list with role/content fields"
        }
    ]


def print_available_datasets():
    """Print information about available datasets"""
    print("\n=== Available Multi-turn Datasets ===")
    for dataset in get_alternative_datasets():
        print(f"\nDataset: {dataset['name']}")
        print(f"Description: {dataset['description']}")
        print(f"Size: {dataset['size']}")
        print(f"Format: {dataset['format']}")
    print("\n" + "="*50)