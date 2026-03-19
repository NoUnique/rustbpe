# rustbpe

[![CI](https://github.com/karpathy/rustbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/karpathy/rustbpe/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rustbpe.svg)](https://pypi.org/project/rustbpe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> The missing tiktoken training code

A lightweight Rust library for training GPT-style BPE tokenizers. The [tiktoken](https://github.com/openai/tiktoken) library is excellent for inference but doesn't support training. The HuggingFace [tokenizers](https://github.com/huggingface/tokenizers) library supports training but carries significant complexity from years of accumulated tokenizer variants. My [minbpe](https://github.com/karpathy/minbpe) library handles both training and inference, but only in Python and not optimized for speed.

**rustbpe** fills this gap: a simple, efficient BPE training implementation in Rust with Python bindings. Train your tokenizer with rustbpe, then export to tiktoken for fast inference.

## Features

- Fast training with parallel processing (rayon)
- GPT-4 style regex pre-tokenization by default
- Special token support: exclude from BPE training, atomic encoding, tiktoken-compatible IDs
- Direct export to tiktoken format
- Python bindings via PyO3
- Batch encoding with automatic parallelization

## Installation

### Python

```bash
pip install rustbpe
```

### From source

```bash
git clone https://github.com/karpathy/rustbpe.git
cd rustbpe
uv venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

## Usage

### Training

```python
import rustbpe

# Create tokenizer and train on your data
tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(
    ["your", "training", "texts", "here"],
    vocab_size=4096
)

# Encode and decode
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)  # "hello world"

# Check vocabulary size
print(tokenizer.vocab_size)  # 4096

# Batch encode (parallel)
all_ids = tokenizer.batch_encode(["text one", "text two", "text three"])
```

### Export to tiktoken

The main use case: train with rustbpe, inference with tiktoken.

```python
import rustbpe
import tiktoken

# Train
tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(open("corpus.txt"), vocab_size=8192)

# Export to tiktoken
enc = tiktoken.Encoding(
    name="my_tokenizer",
    pat_str=tokenizer.get_pattern(),
    mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
    special_tokens={},
)

# Fast inference with tiktoken
ids = enc.encode("hello world")
text = enc.decode(ids)
```

### Special tokens

Special tokens are atomic strings that bypass BPE merging entirely. Register them at training time via the `special_tokens` parameter:

```python
tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(
    texts,
    vocab_size=4096,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
)
```

Two things happen when special tokens are registered:

1. **Excluded from BPE training** — any pre-tokenized chunk that exactly matches a special token string is skipped from merge counting.
2. **IDs assigned after the BPE vocabulary** — special token IDs start at `vocab_size` and increment in list order. They are computed dynamically, so they are always collision-free with merge token IDs.

```python
tokenizer.get_special_tokens()
# {"<s>": 4096, "</s>": 4097, "<pad>": 4098, "<unk>": 4099}

print(tokenizer.vocab_size)  # 4096 (256 bytes + merges only, special tokens not counted)
```

To encode text that may contain special tokens, use `encode_with_special_tokens()` instead of `encode()`. It matches special tokens first (longest match wins), then BPE-encodes the segments between them:

```python
ids = tokenizer.encode_with_special_tokens("<s> hello world </s>")
# [4096, ..., 4097]  — <s> and </s> are atomic; middle text is BPE-encoded

text = tokenizer.decode(ids)  # "<s> hello world </s>"
```

To export to tiktoken with special tokens:

```python
enc = tiktoken.Encoding(
    name="my_tokenizer",
    pat_str=tokenizer.get_pattern(),
    mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
    special_tokens=tokenizer.get_special_tokens(),  # pass the dict directly
)
```

**Behavior notes:**

- `vocab_size` counts only byte tokens + merges. Special tokens are not included.
- `get_mergeable_ranks()` does not include special tokens (tiktoken convention).
- Special tokens must be re-specified on every `train_from_iterator` call. If omitted, the list is cleared.
- After resume training with a larger `vocab_size`, special token IDs shift to the new end of vocabulary. Always call `get_special_tokens()` after training to get the current IDs.

### Custom regex pattern

By default, rustbpe uses the GPT-4 tokenization pattern. You can provide your own:

```python
tokenizer.train_from_iterator(
    texts,
    vocab_size=4096,
    pattern=r"[a-zA-Z]+|[0-9]+|\s+"  # custom pattern
)
```

## API Reference

### `Tokenizer`

| Method | Description |
|--------|-------------|
| `Tokenizer()` | Create a new tokenizer |
| `train_from_iterator(texts, vocab_size, buffer_size=8192, pattern=None, min_frequency=2, special_tokens=None)` | Train on an iterator of strings |
| `encode(text)` | Encode a string to token IDs (ignores special tokens) |
| `encode_with_special_tokens(text)` | Encode a string, treating registered special tokens as atomic units |
| `decode(ids)` | Decode token IDs back to a string (handles special token IDs) |
| `batch_encode(texts)` | Encode multiple strings in parallel |
| `vocab_size` | Property: vocabulary size (256 + number of merges, excludes special tokens) |
| `get_pattern()` | Get the regex pattern used for pre-tokenization |
| `get_mergeable_ranks()` | Get token bytes and ranks for tiktoken export (excludes special tokens) |
| `get_special_tokens()` | Get `{token_string: token_id}` dict for all registered special tokens |
| `get_merges()` | Get all merge rules as `(left_id, right_id, merged_id)` triples |

## Development

### Prerequisites

- Rust: https://rustup.rs/
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup

```bash
git clone https://github.com/karpathy/rustbpe.git
cd rustbpe
uv venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop
```

### Running tests

```bash
# Rust tests (fast, tests core algorithm)
cargo test

# Python tests (requires maturin develop first)
pytest tests/python/ -v -s

# Both
cargo test && pytest tests/python/ -v
```

### Project structure

```
rustbpe/
├── Cargo.toml              # Rust package manifest
├── pyproject.toml          # Python package manifest
├── src/
│   └── lib.rs              # Rust implementation + PyO3 bindings + tests
└── tests/
    └── python/
        └── test_tokenizer.py
```

## How BPE works

Byte Pair Encoding builds a vocabulary iteratively:

1. Start with 256 byte-level tokens (0x00-0xff)
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until reaching target vocabulary size

The result is a vocabulary that efficiently represents common patterns while being able to encode any input.

## LLM Assistance note

I wrote the Python reference code personally and from scratch and I am expert there and understand it fully. I then wrote the Rust code against this implementation with tests for equality. However, I am not a Rust developer by background so I had significant help from ChatGPT and Claude Code Opus 4.5. All the equality tests pass as far as I am aware, but I do apologize if some of the Rust code is not properly arranged, structured, or implemented. Please let me know in Issues/PRs if so and I am happy to adjust the code to make it better.

## License

MIT
