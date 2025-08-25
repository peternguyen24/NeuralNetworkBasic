import tiktoken

enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")

with open('input.txt', 'r') as f:
    text = f.read()

# Character size original
print(f"Input size in chars: {len(text)}")
print(f"Vocab size: {len(set(text))}")

# Token from Tiktoken set
input_tokens = enc.encode(text)
print(f"Input size in tokens: {len(input_tokens)}")
print(f"Vocab size (Tiktoken): {len(set(input_tokens))}")
