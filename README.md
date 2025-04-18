```markdown
# AdOpti — Discrete Prompt‑Optimization (Proof of Concept)

AdOpti shows **how to back‑propagate through token embeddings and project the
result back into discrete space** to craft an adversarial prompt that steers a
causal language model toward a chosen continuation.  
The implementation is intentionally compact and research‑oriented; it is **not
hardened for production use**.

---
```
## Requirements
```bash
pip install torch transformers
```

A CUDA‑capable GPU is recommended.

---

## Quick Start

```bash
python adopti.py  # runs the proof‑of‑concept with GPT‑2
```

By default the script tries to make GPT‑2 continue with the phrase
`" Closed OpenAI"` after a 200‑token learned prefix.

---

## How It Works — Code Walk‑Through
Below, the source is split into logical sections. Inline comments in the
snippets are minimal; the surrounding text explains the intent.

### 1. Imports and Device Selection
```python
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
*Detects GPU availability and selects the execution device.*

---

### 2. The Optimization Entry Point
```python
def find_adversarial_prompt(
    target_text: str,
    model_name: str = "gpt2",
    prompt_length: int = 10,
    num_steps: int = 300,
    lr: float = 1.0,
    inner_steps: int = 1,
    match_threshold: float = 1.0,
    show_steps: bool = False
):
```
*Defines tunable hyper‑parameters for prompt length, learning rate, projection
frequency (`inner_steps`), and early‑stop criteria.*

---

### 3. Model & Tokenizer Setup
```python
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
model.eval()

target_ids = tok.encode(target_text, add_special_tokens=False)
E = model.transformer.wte.weight  # (vocab_size, embed_dim)
```
*Loads the model, ensures a pad token, encodes the target continuation, and
grabs the embedding matrix `E` for nearest‑neighbor projection.*

---

### 4. Prompt Embedding Initialisation
```python
init_ids = [tok.eos_token_id] * prompt_length
p = E[torch.tensor(init_ids, device=DEVICE)].clone().detach()
p.requires_grad_(True)
opt = torch.optim.Adam([p], lr=lr)
```
*Starts with `prompt_length` repetitions of the EOS embedding and attaches an
Adam optimizer to those continuous vectors.*

---

### 5. Optimisation Loop
```python
for step in range(1, num_steps + 1):
    opt.zero_grad()
    inp, lbl = build(p)          # concatenate prompt + target
    loss = model(inputs_embeds=inp, labels=lbl).loss
    loss.backward()
    opt.step()
```
*Back‑propagates through the model to make the prompt better predict the target
tokens.*

---

### 6. Discrete Projection and Early Stopping
```python
if step % inner_steps == 0:
    with torch.no_grad():
        # Project each prompt vector to its nearest real token embedding
        ids = ((p**2).sum(1, keepdim=True)
               + norms.unsqueeze(0)
               - 2 * (p @ E.T)).argmin(1)
        p[:] = E[ids]
```
*Every `inner_steps`, the continuous prompt is **snapped back to discrete
tokens** by nearest‑neighbor search in embedding space, enabling pure‑token
generation.*

An early exit occurs when the generated continuation matches the target above
`match_threshold`.

---

### 7. Final Rounding and Return
```python
final_ids = ((p**2).sum(1, keepdim=True)
             + norms.unsqueeze(0)
             - 2 * (p @ E.T)).argmin(1).cpu().tolist()
final_prompt = tok.decode(final_ids, clean_up_tokenization_spaces=False)
return final_ids, final_prompt
```
*Rounds the best continuous prompt seen and returns both IDs and decoded
string.*

---

### 8. Script Entrypoint
```python
if __name__ == "__main__":
    ids, prompt = find_adversarial_prompt(
        " Closed OpenAI",
        prompt_length=200,
        num_steps=1000,
        lr=1.2,
        inner_steps=2,
        match_threshold=1.0,
    )
    print("→ Prompt IDs:", ids)
    print("→ Decoded prompt:", prompt)
```
*Provides a runnable example demonstrating how to craft a 200‑token adversarial
prefix for GPT‑2.*

---

## License and Intended Use
This repository is released under the MIT License and is **provided solely as a
proof of concept for robustness research**.  
Misuse to produce harmful or disallowed content is discouraged.
```
