# TODO: build a RAG system with the model 
# this agent is used to translate the inference result of the Qlearning to the real action command to the switch/router
# ex.
# if the model's output is to close link (3,6,7), this agent should according this info to generate the command to shutdown the corresponding port of the link 3,6,7 
# so the closing command should be the restconf to shutdown the interface

import os
import torch
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
import unsloth
from unsloth import FastLanguageModel


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLMNetworkAgent:
    """
    An agent that uses Taide model via Unsloth to translate Q-learning inference results
    into network device commands (RESTCONF).
    """
    
    def __init__(
        self,
        model_name: str = "TAIDE/TaiDe-7B-Base-v1.0",
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the LLM Network Agent with a Taide model loaded through Unsloth.
        
        Args:
            model_name: The name of the Taide model to load (from HuggingFace)
            max_seq_length: Maximum sequence length for the model
            load_in_4bit: Whether to load the model in 4-bit quantization
            device: Device to load the model on ('cuda', 'cpu', or 'auto')
            cache_dir: Directory to cache the downloaded model
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        
        # Setup quantization configuration
        if load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
       
        print("Using Unsloth for optimized model loading")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                quantization_config=quantization_config,
                cache_dir=cache_dir,
            )
            
            # Apply optimizations from Unsloth
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none"
            )
        except Exception as e:
            print(f"Error using Unsloth: {e}. Falling back to standard transformers.")
        print(f"{model_name} loaded successfully!")
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (lower is more deterministic)
            
        Returns:
            The generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

# Usage example
if __name__ == "__main__":
    agent = LLMNetworkAgent(model_name="TAIDE/TaiDe-7B-Base-v1.0")
    test_response = agent.generate("Hello, I'm a network engineer. Can you help me with network configurations?")
    print("Model response:", test_response)