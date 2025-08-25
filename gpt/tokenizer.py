import tiktoken
import sentencepiece as spm

enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")

with open('input.txt', 'r') as f:
    text = f.read()

# Character size original
print(f"Input size in chars: {len(text)}")
print(f"Vocab size: {len(set(text))}")

# Token from SentencePiece lib
import os
options = dict(
  # input spec
  input="input.txt",
  input_format="text",
  # output spec
  model_prefix="tok600", # output filename prefix
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=600,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)
spm.SentencePieceTrainer.train(**options)
sp = spm.SentencePieceProcessor()
sp.load("tok600.model")

vocab = [(sp.id_to_piece(idx) + "->" + str(idx)) for idx in range(sp.get_piece_size())]
print("Input size in tokens (SentencePiece):", len(sp.encode(text)))
print("Vocab size (SentencePiece):", sp.get_piece_size())
print(vocab)
