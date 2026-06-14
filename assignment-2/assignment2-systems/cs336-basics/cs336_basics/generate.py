import torch

from cs336_basics.model import Transformer
from cs336_basics.utils import load_checkpoint, softmax
from cs336_basics.tokenization import Tokenizer

src="/Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/artifacts/checkpoints/step_1000.pt"
config = {"vocab_size": 10000,
           "context_length": 64, 
           "num_layers": 2,
           "d_model": 1024,
           "num_heads": 4, 
           "d_ff": 2688, 
           "theta": 1e5, 
           "device": "cpu"}

def decoding(prompt: str, max_nb_tokens: int = 200, threshold: float = 0.8, temperature: float = 0.8) -> str:
    model = Transformer(**config)
    load_checkpoint(model=model, src=src)

    tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/tinystories/vocab.json", 
        merges_filepath="artifacts/tinystories/merges.txt", 
        special_tokens=["<|endoftext|>"])
    
    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer.encode(prompt)
        tokens = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0)
        eot = tokenizer.vocab_inverse[b'<|endoftext|>'] 
        nb_tokens = 0
        while nb_tokens < max_nb_tokens:
            truncated = tokens[:, -context_length:]
            logits = model(truncated)
            last_logits = logits[:, -1, :]
            probs = softmax(last_logits, dim=-1, temperature=temperature)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)

            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= threshold
            pad = torch.ones_like(mask[..., :1])
            mask = torch.cat([pad, mask[..., :-1]], dim=-1)
            filtered_probs = probs * mask

            sample_sorted_position = torch.multinomial(filtered_probs, 1).squeeze(-1)
            next_id = sorted_idx.gather(dim=-1, index=sample_sorted_position.unsqueeze(0)).squeeze(-1)
            if next_id.item() == eot:
                break
            tokens = torch.cat([tokens, next_id.unsqueeze(0)], dim=1)
            nb_tokens += 1
    generated_ids = tokens[0, len(encoded_input):].tolist()
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    output = decoding(prompt= "tell me a story", max_nb_tokens=100, threshold=0.8, temperature=1.2)
    print(output)
