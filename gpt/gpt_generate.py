import torch
import gpt_model

model = gpt_model.BigramLanguageModel()
model.load_state_dict(torch.load('model.pth'))

m = model.to(gpt_model.device)

context = torch.zeros((1, 1), dtype=torch.long, device=gpt_model.device)
with open('output.txt', 'w') as f:
    outs = gpt_model.decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    f.write(outs)

print("Finished writing to output.txt")