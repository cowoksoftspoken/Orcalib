import pytest
import orca
import orca.nn as nn
import orca.optim as optim

@pytest.fixture
def cpu_device():
    """Fixture to provide CPU device string."""
    return "cpu"

@pytest.fixture
def gpu_device():
    """Fixture to provide GPU device string."""
    return "gpu"

@pytest.fixture
def simple_model():
    """Fixture to construct a simple 2->2 Linear model."""
    return nn.Sequential(nn.Linear(2, 2))

@pytest.mark.parametrize("shape,expected_len", [
    ([2, 3], 6),
    ([1, 4], 4),
    ([5], 5),
])
def test_tensor_creation(shape, expected_len):
    """Verify tensor allocation shape and zero initialization using pytest parametrization."""
    x = orca.Tensor.zeros(shape)
    assert x.shape == shape
    assert len(x.to_list()) == expected_len
    assert all(val == 0.0 for val in x.to_list())

@pytest.mark.parametrize("a_val,b_val,op", [
    ([1.0, 2.0], [3.0, 4.0], "add"),
    ([2.0, 3.0], [4.0, 5.0], "sub"),
    ([1.0, 2.0], [3.0, 4.0], "mul"),
])
def test_tensor_ops(a_val, b_val, op):
    """Test elementary binary arithmetic operations on parameterized inputs."""
    a = orca.Tensor.from_list(a_val, shape=[1, len(a_val)])
    b = orca.Tensor.from_list(b_val, shape=[1, len(b_val)])
    
    if op == "add":
        c = a + b
        assert c.to_list() == [x + y for x, y in zip(a_val, b_val)]
    elif op == "sub":
        c = a - b
        assert c.to_list() == [x - y for x, y in zip(a_val, b_val)]
    elif op == "mul":
        c = a * b
        assert c.to_list() == [x * y for x, y in zip(a_val, b_val)]

def test_autograd_basic():
    """Verify standard automatic differentiation backward gradient accumulation."""
    x = orca.Tensor.from_list([2.0], shape=[1, 1], requires_grad=True)
    y = x * x  # y = x^2
    y.backward()
    
    assert x.grad() is not None
    # dy/dx = 2 * x = 4.0
    assert abs(x.grad().to_list()[0] - 4.0) < 1e-5

def test_shape_mismatch_exception():
    """Verify that backend shape mismatches raise a ValueError using pytest.raises."""
    a = orca.Tensor.from_list([1.0, 2.0], shape=[1, 2])
    b = orca.Tensor.from_list([1.0, 2.0, 3.0], shape=[1, 3])
    with pytest.raises(ValueError):
        _ = a + b

def test_linear_layer(simple_model):
    """Test linear forward pass with a shared model fixture."""
    x = orca.Tensor.ones([1, 2])
    out = simple_model(x)
    assert out.shape == [1, 2]

def test_crossentropy_loss():
    """Test CrossEntropyLoss stability and output range."""
    loss_fn = nn.CrossEntropyLoss()
    pred = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    target = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss = loss_fn(pred, target)
    assert loss.to_list()[0] > 0.0

@pytest.mark.parametrize("optimizer_class", [
    optim.SGD,
    optim.Adam,
    optim.AdamW,
])
def test_optimizers(simple_model, optimizer_class):
    """Parametrized verification of SGD, Adam, and AdamW optimizer parameter updates."""
    if optimizer_class == optim.SGD:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    elif optimizer_class == optim.AdamW:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.01, weight_decay=0.01)
    else:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.01)
        
    x = orca.Tensor.ones([1, 2])
    y = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss_fn = nn.MSELoss()
    
    # Save initial weights
    params = list(simple_model.parameters())
    init_weights = params[0].tensor.to_list()
    
    # Update
    optimizer.zero_grad()
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    
    # Verify weights actually updated
    updated_weights = params[0].tensor.to_list()
    assert init_weights != updated_weights

def test_numerical_gradients():
    """Check analytical autograd correctness using numerical finite differences."""
    x_val = 2.0
    eps = 1e-4

    # Analytical
    x = orca.Tensor.from_list([x_val], shape=[1], requires_grad=True)
    y = x * x * x + x * x * orca.Tensor.scalar(2.0) + x * orca.Tensor.scalar(5.0)
    y.backward()
    grad_analytical = x.grad().to_list()[0]

    # Numerical f(x + eps)
    x_plus = orca.Tensor.from_list([x_val + eps], shape=[1])
    y_plus = x_plus * x_plus * x_plus + x_plus * x_plus * orca.Tensor.scalar(2.0) + x_plus * orca.Tensor.scalar(5.0)
    
    # f(x - eps)
    x_minus = orca.Tensor.from_list([x_val - eps], shape=[1])
    y_minus = x_minus * x_minus * x_minus + x_minus * x_minus * orca.Tensor.scalar(2.0) + x_minus * orca.Tensor.scalar(5.0)

    grad_numerical = (y_plus.to_list()[0] - y_minus.to_list()[0]) / (2.0 * eps)

    assert abs(grad_analytical - 25.0) < 1e-2
    assert abs(grad_analytical - grad_numerical) < 1e-2


@pytest.mark.parametrize("scheduler_class,kwargs", [
    (optim.StepLR, {"step_size": 2, "gamma": 0.5}),
    (optim.CosineAnnealingLR, {"T_max": 10, "eta_min": 0.001}),
    (optim.LinearWarmup, {"warmup_epochs": 5}),
])
def test_lr_schedulers(simple_model, scheduler_class, kwargs):
    """Verify learning rate schedulers update optimizer learning rates correctly."""
    optimizer = optim.SGD(simple_model.parameters(), lr=0.1)
    scheduler = scheduler_class(optimizer, **kwargs)
    
    initial_lr = optimizer.lr
    
    # Step the scheduler
    scheduler.step()
    first_step_lr = optimizer.lr
    
    if scheduler_class == optim.StepLR:
        # Step 1: last_epoch=1, step_size=2 -> no change
        assert first_step_lr == initial_lr
        scheduler.step() # last_epoch=2 -> lr *= gamma
        assert optimizer.lr == initial_lr * 0.5
    elif scheduler_class == optim.CosineAnnealingLR:
        # Should change because of cosine decay
        assert first_step_lr < initial_lr
    elif scheduler_class == optim.LinearWarmup:
        # At epoch 0 (initially) lr is 0.1, at first step (last_epoch=1) it should be 0.1 * 1/5 = 0.02
        assert first_step_lr == pytest.approx(0.02)


def test_linear_numerical_gradient():
    """Verify Linear layer gradients using numerical finite differences."""
    # Simple model: y = x @ W
    # where x is [1, 2], W is [2, 1]. Out is [1, 1].
    x_val = [1.5, -2.0]
    w_val = [0.8, 1.2]
    eps = 1e-4
    
    # Analytical backward pass
    x = orca.Tensor.from_list(x_val, shape=[1, 2])
    W = orca.Tensor.from_list(w_val, shape=[2, 1], requires_grad=True)
    
    y = x @ W
    y.backward()
    grad_W = W.grad().to_list()
    
    # Numerical backward pass
    # f(W + eps) for W[0]
    W_plus = orca.Tensor.from_list([w_val[0] + eps, w_val[1]], shape=[2, 1])
    y_plus = (x @ W_plus).to_list()[0]
    
    # f(W - eps) for W[0]
    W_minus = orca.Tensor.from_list([w_val[0] - eps, w_val[1]], shape=[2, 1])
    y_minus = (x @ W_minus).to_list()[0]
    
    grad_num_0 = (y_plus - y_minus) / (2.0 * eps)
    
    # Compare
    assert abs(grad_W[0] - grad_num_0) < 1e-2
    assert abs(grad_W[0] - x_val[0]) < 1e-3


def test_grad_scaler(simple_model):
    """Verify GradScaler scales loss, steps optimizer, and updates scales correctly."""
    from orca import GradScaler
    scaler = GradScaler(init_scale=1024.0, growth_interval=2)
    optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
    
    x = orca.Tensor.ones([1, 2])
    y = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss_fn = nn.MSELoss()
    
    # 1. Scale loss
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    scaled_loss = scaler.scale_loss(loss)
    
    assert scaled_loss.to_list()[0] == pytest.approx(loss.to_list()[0] * 1024.0)
    
    # 2. Backward
    optimizer.zero_grad()
    scaled_loss.backward()
    
    # 3. Step
    scaler.step(optimizer)
    scaler.update()
    
    # Scale should remain same since growth interval is 2
    assert scaler.scale == 1024.0
    
    # Step again to trigger growth
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    scaled_loss = scaler.scale_loss(loss)
    optimizer.zero_grad()
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Scale should have grown by growth_factor (2.0)
    assert scaler.scale == 2048.0


