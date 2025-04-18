
# AdOpti — Discrete Prompt‑Optimization (Proof of Concept)

AdOpti shows **how to back‑propagate through token embeddings and project the
result back into discrete space** to craft an adversarial prompt that steers a
causal language model toward a chosen continuation.  
The implementation is intentionally compact and research‑oriented; it is **not
hardened for production use**.

---
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

<details>
<summary>Sample console output (click to expand)</summary>

<pre>
Optimizing...
Step 2/1000 | Gen:  French French French | Loss: 11.1554
Step 4/1000 | Gen:  Frenchmen French | Loss: 8.5308
Step 6/1000 | Gen:  Closed FrenchAI | Loss: 7.2439
Step 8/1000 | Gen:  Closed Closed Closed | Loss: 3.8840
Step 10/1000 | Gen:  Closed Closed Closed | Loss: 3.9848
Step 12/1000 | Gen:  German German German | Loss: 1.5244
Step 14/1000 | Gen:  Openair  | Loss: 2.5160
Step 16/1000 | Gen:  OpenAI Open | Loss: 2.4235
Step 18/1000 | Gen:  Closed OpenAI | Loss: 2.4046
✔️ Matched at step 18!
→ Prompt IDs: [44952, 33524, 470, 44014, 19310, 29855, 39906, 24056, 47846, 8384, 49639, 15966, 5533, 9430, 2583, 18844, 18844, 35395, 49843, 46191, 18452, 622, 764, 46328, 46328, 11651, 22135, 46328, 39534, 37045, 20253, 44128, 34751, 8546, 8146, 22398, 22398, 22398, 22398, 22398, 22398, 48366, 48366, 45201, 42543, 45201, 45786, 46666, 31058, 35372, 35372, 45201, 45201, 45201, 48853, 20801, 27049, 4115, 20801, 45144, 43024, 26575, 20801, 34877, 49190, 35523, 49817, 15166, 18161, 5061, 140, 140, 140, 28819, 28819, 140, 28819, 28819, 28819, 28819, 46812, 16879, 27090, 46437, 30010, 34127, 34127, 42849, 8557, 34127, 36218, 15920, 36218, 7551, 4348, 1649, 2909, 2909, 4721, 2941, 2941, 4863, 36218, 35266, 27622, 39906, 35912, 13958, 27467, 35912, 36218, 27467, 36218, 50113, 18, 6681, 50256, 34881, 39421, 13926, 1567, 4275, 4275, 4275, 4275, 4275, 11505, 21017, 12635, 45528, 48585, 22686, 49880, 11260, 9918, 5094, 16529, 28781, 20185, 27634, 20185, 1391, 49182, 44051, 46114, 30109, 23330, 23330, 9552, 26785, 12662, 49485, 15575, 2698, 12701, 12855, 33404, 4518, 16921, 47745, 8099, 20185, 44420, 28147, 41553, 22741, 25940, 42748, 9403, 38000, 2266, 23606, 27570, 371, 48395, 48395, 48373, 2677, 23441, 44715, 50030, 27207, 30550, 16512, 16512, 4946, 47594, 17735, 45606, 26591, 43061, 4141, 4141, 3594, 4141, 4141, 49625, 958, 27772, 8008]
→ Decoded prompt: foreseenanamo't nonprofits PT GingrichEStream enduring indemn mainlyaggressiveanyahu Islamicamps hostPacPacbitcoin▬▬ disorderlyaicru . .) .)ONE ." .) CoveATIVE airst avalanchesweetosaoricicansicansicansicansicansicans◼◼ conduc Seym conduc▄obylierrekayakaya conduc conduc conduc CroatianedIn RegisteredibraryedIn "{ Retrieved quiredInetsy inappropriatelyThumbnail Hitchcockо."[ Edition���utterstockutterstock�utterstockutterstockutterstockutterstockINTONHERtesyPREomezpoliticspolitics Rousse spiritualpolitics AGA%), AGAzonaoday Whenroughtrought openedccordingccordingFrom AGA� DucksEStream ALSO ×Enjoy ALSO AGAEnjoy AGA.''.3 withd<|endoftext|> CNBC APPLICeconomaugh (@ (@ (@ (@ (@Open### Monster Adin���         VIEW Caption Anderson Public ----------------------------------------------------------------tlAI ArduinoAI { MGM Clicker Cyborg[[_{_{ AI Kag Citiz joystick puzz laun patent laser mourning Bluevidmeet PopAI Cortex Angular labyrinth Controller Hiro Kyr seed Rag red symm dissatisf R Rao Rao tul King cubeakov glared Identity Closed closure closure Open ak Ny SchnalphaSalt French French English French Frenchshutair Guant��
</pre>

</details>
*Yes if you run GPT with that prompt it will generate the tokens "Closed OpenAI"

---

## License and Intended Use
This repository is released under the MIT License and is **provided solely as a
proof of concept for robustness research**.  
Misuse to produce harmful or disallowed content is discouraged.
```
