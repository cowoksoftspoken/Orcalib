import pytest
import orca
import orca.nn as nn
from orca.zoo import ResNet18, BERT, GPT2

def test_resnet18_shapes():
    """Verify ResNet-18 model architecture parameters and forward output shapes."""
    model = ResNet18(num_classes=10)
    model.eval()
    
    # 3-channel image batch [2, 3, 32, 32] (scaled down for speed)
    x = orca.Tensor.randn([2, 3, 32, 32])
    out = model(x)
    
    assert out.shape == [2, 10]
    
    # Validate some parameter shapes
    params = list(model.parameters())
    # first conv weights should be [64, 3, 7, 7]
    assert params[0].tensor.shape == [64, 3, 7, 7]
    # final classifier weights should be [512, 10]
    assert params[-2].tensor.shape == [512, 10]

def test_gpt2_shapes():
    """Verify GPT2 model architecture parameters and forward output shapes."""
    model = GPT2(vocab_size=1000, max_position_embeddings=128, embed_dim=64, num_heads=4, num_layers=2)
    model.eval()
    
    # [batch=2, seq_len=8, vocab_size=1000] (one-hot input)
    x = orca.Tensor.randn([2, 8, 1000])
    out = model(x)
    
    assert out.shape == [2, 8, 1000]
    
    params = list(model.parameters())
    # Word embeddings wte.weight should be [1000, 64]
    assert params[0].tensor.shape == [1000, 64]
    # Position embeddings wpe.weight should be [128, 64]
    assert params[1].tensor.shape == [128, 64]

def test_bert_shapes():
    """Verify BERT model architecture parameters and forward output shapes."""
    model = BERT(vocab_size=1000, max_position_embeddings=128, type_vocab_size=2, hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
    model.eval()
    
    # [batch=2, seq_len=8, vocab_size=1000]
    x = orca.Tensor.randn([2, 8, 1000])
    out = model(x)
    
    assert out.shape == [2, 8, 64]
    
    params = list(model.parameters())
    # Word embeddings should be [1000, 64]
    assert params[0].tensor.shape == [1000, 64]
    # Position embeddings should be [128, 64]
    assert params[1].tensor.shape == [128, 64]

def test_hub_loader_unpretrained():
    """Verify load with pretrained=False creates architectures matching default sizes."""
    m_gpt2 = orca.hf.load("gpt2", pretrained=False)
    assert isinstance(m_gpt2, GPT2)
    assert m_gpt2.vocab_size == 50257
    assert m_gpt2.embed_dim == 768
