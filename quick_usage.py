import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "dmis-lab/self-biorag-7b-olaph" # ["mistralai/Mistral-7B-v0.1", "BioMistral/BioMistral-7B", "meta-llama/Llama-2-7b-hf", "dmis-lab/selfbiorag_7b", "epfl-llm/meditron-7b"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    query = "Can a red eye be serious?"

    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, do_sample=False, top_p=1.0).to(device)
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    print ("Model prediction: ", response[len(query):].strip())
            

if __name__ == "__main__":
    main()