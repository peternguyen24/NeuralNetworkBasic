0. Tokenize the input set using SentencePiece, by runing `py tokenizer.py`
1. Create and train model by running `py gpt_training.py`.
    * Model will be trained on data from `input.txt`
    * Model will be written to disk as `model.pth`

2. Run `gpt_generate.py` to generate 5000 words from model. 
    * Output will be written to `output.txt`

`input.txt` and `output.txt` are uploaded to this directory for reference. However, swapping `input.txt` with different content is okay, and will result in a new trained model thus different `output.txt`. 

NOTE: This mode is a currently a bit overfitting, as the last training step is `Step 9800, Train Loss: 2.2848, Val Loss: 2.8769`