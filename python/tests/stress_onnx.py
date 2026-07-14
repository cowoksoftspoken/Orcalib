import orca
import orca.nn as nn
import orca.optim as optim
import os
import time

class DeepMLP(nn.Module):
    def __init__(self, in_dim=100, hidden_dim=256, out_dim=10, depth=5):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim, bias=True))
        self.seq = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.seq(x)

def run_onnx_stress_test(device="cpu"):
    print(f"\n--- [Stress Test] ONNX Exporter/Importer ({device}) ---")
    
    # 1. Instantiate a deep model
    model = DeepMLP(depth=6)
    model.to(device)
    model.eval()
    
    # 2. Large input shape [512, 100]
    x = orca.Tensor.randn([512, 100]).to(device)
    
    # Forward original
    t0 = time.time()
    y_orig = model(x)
    t1 = time.time()
    print(f"  Original forward pass took {(t1 - t0)*1000:.2f}ms.")
    
    # 3. Export to ONNX
    onnx_path = f"stress_mlp_{device}.onnx"
    t0 = time.time()
    orca.onnx.export_onnx(model, x, onnx_path)
    t1 = time.time()
    print(f"  Export to ONNX took {(t1 - t0)*1000:.2f}ms.")
    
    # 4. Import model back
    t0 = time.time()
    imported_model = orca.onnx.import_onnx(onnx_path)
    t1 = time.time()
    print(f"  Import from ONNX took {(t1 - t0)*1000:.2f}ms.")
    
    imported_model.to(device)
    imported_model.eval()
    
    # 5. Run forward on imported model and compare
    t0 = time.time()
    y_imp = imported_model(x)
    t1 = time.time()
    print(f"  Imported forward pass took {(t1 - t0)*1000:.2f}ms.")
    
    # Check shape and value match
    assert y_orig.shape == y_imp.shape
    y_orig_list = y_orig.to_list()
    y_imp_list = y_imp.to_list()
    for a, b in zip(y_orig_list, y_imp_list):
        assert abs(a - b) < 1e-4
    print("  [PASS] Forward output consistency matches exactly within 1e-4 tolerance.")
    
    # 6. Stress test training loop on imported model for 50 steps
    imported_model.train()
    for p in imported_model.parameters():
        p.tensor.require_grad()
        
    optimizer = optim.Adam(imported_model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    target = orca.Tensor.ones(y_imp.shape).to(device)
    
    print("  Running 50 optimization steps on the imported model...")
    t0 = time.time()
    for step in range(50):
        optimizer.zero_grad()
        pred = imported_model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
    t1 = time.time()
    print(f"  Completed 50 backpropagation training steps in {t1 - t0:.2f}s.")
    print(f"  Final training step loss: {loss.to_list()[0]:.6f}")
    
    # Clean up file
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    print(f"  [PASS] ONNX stress test on {device} completed successfully.")

if __name__ == "__main__":
    print("==================================================")
    print("STARTING ONNX EXPORT/IMPORT STRESS TEST")
    print("==================================================")
    
    run_onnx_stress_test("cpu")
    run_onnx_stress_test("gpu")
    
    print("\n==================================================")
    print("ONNX EXPORT/IMPORT STRESS TEST PASSED SUCCESSFULLY!")
    print("==================================================")
