import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Mock objects for LLM if it cannot be loaded
class MockTokenizer:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    def decode(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    @property
    def eos_token_id(self):
        return 0

class MockModel:
    def generate(self, *args, **kwargs):
        raise NotImplementedError("LLM model is not loaded. Please ensure LLM setup is correct.")
    def eval(self):
        pass

# Global variables for models to avoid reloading
_llm_tokenizer = None
_llm_model = None
_llm_device = None

def load_llm_and_tokenizer_base(model_name: str, device: torch.device):
    """
    Loads the Large Language Model (LLM) and its tokenizer for base LLM usage.
    Handles potential import errors for large model dependencies.
    """
    global _llm_tokenizer, _llm_model, _llm_device
    if _llm_tokenizer is None or _llm_model is None:
        print(f"Attempting to load LLM model and tokenizer: {model_name} on {device}")
        _llm_device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, # Adjust based on GPU support
                device_map="auto",         # Recommended for large models, requires `accelerate`
            )
            _llm_model.eval() # Set model to evaluation mode for inference
            print(f"LLM '{model_name}' loaded successfully on {_llm_model.device}.")
        except ImportError as e:
            print(f"Error importing LLM libraries (e.g., transformers, accelerate, bitsandbytes): {e}")
            print("Falling back to mock LLM. Please install necessary libraries for full functionality.")
            _llm_tokenizer = MockTokenizer()
            _llm_model = MockModel()
        except Exception as e:
            print(f"An unexpected error occurred during LLM loading: {e}")
            print("Falling back to mock LLM. This might be due to insufficient VRAM or other setup issues.")
            _llm_tokenizer = MockTokenizer()
            _llm_model = MockModel()

    return _llm_tokenizer, _llm_model

def build_conversation_prompt(question: str, chat_history: list[tuple[str, str]]):
    """
    Builds a prompt including previous conversation turns for the LLM.
    `chat_history` is a list of (user_message, llm_response) tuples.
    """
    system_prompt = "You are a helpful cybersecurity expert. Answer questions based on the conversation history."
    
    prompt_parts = [f"### System:\\n{system_prompt}\\n"]
    
    for user_msg, llm_resp in chat_history:
        prompt_parts.append(f"### User:\\n{user_msg}\\n")
        prompt_parts.append(f"### Assistant:\\n{llm_resp}\\n")
        
    prompt_parts.append(f"### User:\\n{question}\\n### Assistant:\\n")
    
    return "".join(prompt_parts)

def generate_response_without_rag(question: str, chat_history: list[tuple[str, str]], llm_model_name: str, device: torch.device):
    """
    Generates a response from the LLM without RAG, considering chat history.
    """
    llm_tokenizer, llm_model = load_llm_and_tokenizer_base(llm_model_name, device)

    if isinstance(llm_model, MockModel):
        print("\\n--- LLM Not Loaded ---")
        print("Skipping generation as the LLM model is not loaded. Please address LLM loading issues.")
        print("--------------------\\n")
        return "LLM not loaded. Cannot generate response without RAG."

    prompt = build_conversation_prompt(question, chat_history)

    print("\\n--- Generated Prompt (for LLM - without RAG) ---")
    print(prompt)
    print("--------------------------------------------------\\n")

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=250, # A bit more verbose as it has no external context
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_return_sequences=1
        )

    response = llm_tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    from config import LLM_MODEL_NAME, DEVICE, get_optimal_device

    print("Running LLM Without RAG (Interactive Chat Mode)...")
    # Ensure device is correctly set for the interactive session
    DEVICE = get_optimal_device()
    
    chat_history = [] # Stores (user_message, llm_response) tuples

    while True:
        user_question = input("\\nUser: ")
        if user_question.lower() == 'quit':
            break

        try:
            llm_response = generate_response_without_rag(
                user_question,
                chat_history,
                LLM_MODEL_NAME,
                DEVICE
            )
            print("\\nAssistant (without RAG):", llm_response)
            chat_history.append((user_question, llm_response))
            
        except NotImplementedError as e:
            print(f"Error: {e}")
            break # Exit if LLM cannot be loaded
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\\nExiting LLM without RAG chat.")