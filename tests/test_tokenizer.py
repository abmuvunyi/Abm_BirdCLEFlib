import pytest
import os
import tempfile
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from data.tokenizer import H4Tokenizer

@pytest.fixture
def dummy_tokenizer_file():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        tokenizer.save(f.name)
        yield f.name
    os.remove(f.name)

def test_tokenizer_init(dummy_tokenizer_file):
    token_map = {'char': dummy_tokenizer_file}
    tokenizer = H4Tokenizer(token_map=token_map, token_type='char', validate=False)
    assert tokenizer.token_type == 'char'

def test_tokenizer_invalid_type(dummy_tokenizer_file):
    token_map = {'char': dummy_tokenizer_file}
    with pytest.raises(ValueError):
        H4Tokenizer(token_map=token_map, token_type='invalid', validate=False)
