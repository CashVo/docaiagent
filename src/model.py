from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM

# Load the Llama model
def load_llama_model(model_id):    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer=tokenizer
        )

    return { 
        "model": llm, 
        "tokenizer": tokenizer 
    }
