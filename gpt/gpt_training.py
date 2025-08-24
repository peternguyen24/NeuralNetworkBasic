import torch
import gpt_model

model = gpt_model.BigramLanguageModel()
m = model.to(gpt_model.device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Training
for iter in range(gpt_model.max_iters): 

    # every once in a while (modulus eval_iters), we evaluate the loss on train and val sets
    if iter % gpt_model.eval_iters == 0:
        train_loss = gpt_model.estimate_loss(model = m)
        print(f"Step {iter}, Train Loss: {train_loss['train']:.4f}, Val Loss: {train_loss['val']:.4f}")

    # sample a batch of data
    xb, yb = gpt_model.get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), 'model.pth')