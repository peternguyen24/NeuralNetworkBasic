import torch
import gpt_model
import sentencepiece as spm

model = gpt_model.BigramLanguageModel()
model.load_state_dict(torch.load('model.pth'))

sp = spm.SentencePieceProcessor()
sp.load("tok600.model")

m = model.to(gpt_model.device)

context = torch.zeros((1, 1), dtype=torch.long, device=gpt_model.device)
context[0,0] = sp.encode("A")[0] # Start with token A
outs = m.generate(context, max_new_tokens=5000)

# Write outs to output.txt using UTF-8 encoding
with open("output.txt", "w", encoding="utf-8") as f:
	f.write(str(outs))  

print("Finished writing to output.txt")