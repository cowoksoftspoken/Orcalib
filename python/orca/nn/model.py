from .module import Module
import time

class Model(Module):
    """
    A high-level Model abstraction, similar to Keras, that provides orchestration
    methods like `compile`, `fit`, `evaluate`, and `predict`.
    """
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.loss_fn = None
        self.metrics = None

    def compile(self, optimizer, loss, metrics=None):
        """
        Configures the model for training.
        
        Args:
            optimizer: String alias (e.g., 'sgd', 'adam') or an Optimizer instance.
            loss: String alias (e.g., 'crossentropy', 'mse') or a Loss module instance.
            metrics: List of metrics to be evaluated by the model during training and testing.
        """
        # Resolve Loss
        if isinstance(loss, str):
            if loss.lower() in ['crossentropy', 'ce']:
                from .loss import CrossEntropyLoss
                self.loss_fn = CrossEntropyLoss()
            elif loss.lower() == 'mse':
                from .loss import MSELoss
                self.loss_fn = MSELoss()
            else:
                raise ValueError(f"Unknown loss alias: {loss}")
        else:
            self.loss_fn = loss

        # Resolve Optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'sgd':
                from orca.optim.sgd import SGD
                self.optimizer = SGD(self.parameters())
            elif optimizer.lower() == 'adam':
                from orca.optim.adam import Adam
                self.optimizer = Adam(self.parameters())
            else:
                raise ValueError(f"Unknown optimizer alias: {optimizer}")
        else:
            self.optimizer = optimizer
            
        self.metrics = metrics or []
        
    def _compute_metrics(self, predictions, targets):
        import numpy as np
        results = {}
        for m in self.metrics:
            if m == 'accuracy':
                p_list = predictions.to_list()
                t_list = targets.to_list()
                # Fast numpy convert
                p_arr = np.array(p_list)
                t_arr = np.array(t_list)
                if len(p_arr.shape) == 2 and p_arr.shape[1] > 1:
                    # Categorical / One-hot
                    pred_labels = np.argmax(p_arr, axis=1)
                    if len(t_arr.shape) == 2 and t_arr.shape[1] > 1:
                        target_labels = np.argmax(t_arr, axis=1)
                    else:
                        target_labels = t_arr
                    acc = np.mean(pred_labels == target_labels)
                else:
                    # Binary or regression (threshold 0.5)
                    pred_labels = (p_arr > 0.5).astype(np.float32)
                    acc = np.mean(pred_labels == t_arr)
                results['accuracy'] = float(acc)
        return results

    def fit(self, x, y=None, batch_size=32, epochs=1, validation_data=None, shuffle=True):
        """
        Trains the model for a fixed number of epochs.
        If `x` is a DataLoader or iterator, `y` can be None.
        """
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("You must compile the model before training.")
            
        is_iterator = hasattr(x, '__iter__') and not isinstance(x, (list, tuple))
        
        for epoch in range(epochs):
            self.train()
            print(f"Epoch {epoch+1}/{epochs}")
            
            epoch_loss = 0.0
            epoch_metrics = {m: 0.0 for m in self.metrics}
            start_time = time.time()
            
            if is_iterator:
                step = 0
                for batch_x, batch_y in x:
                    # Forward pass
                    self.optimizer.zero_grad()
                    predictions = self(batch_x)
                    loss_val = self.loss_fn(predictions, batch_y)
                    
                    # Backward pass
                    loss_val.backward()
                    self.optimizer.step()
                    
                    val = loss_val.to_list()
                    epoch_loss += val[0] if isinstance(val, list) else val
                    
                    # Compute metrics
                    batch_metrics = self._compute_metrics(predictions, batch_y)
                    for m, v in batch_metrics.items():
                        epoch_metrics[m] += v
                        
                    step += 1
                    
                    metrics_str = f" - loss: {epoch_loss / step:.4f}"
                    for m in self.metrics:
                        metrics_str += f" - {m}: {epoch_metrics[m] / step:.4f}"
                        
                    print(f"\rStep {step}{metrics_str}", end="")
                print(f" - {(time.time() - start_time):.1f}s")
            else:
                self.optimizer.zero_grad()
                predictions = self(x)
                loss_val = self.loss_fn(predictions, y)
                loss_val.backward()
                self.optimizer.step()
                
                val = loss_val.to_list()
                epoch_loss = val[0] if isinstance(val, list) else val
                batch_metrics = self._compute_metrics(predictions, y)
                
                metrics_str = f" - loss: {epoch_loss:.4f}"
                for m in self.metrics:
                    metrics_str += f" - {m}: {batch_metrics[m]:.4f}"
                    
                print(f"\rStep 1/1{metrics_str} - {(time.time() - start_time):.1f}s")
            
            if validation_data is not None:
                val_x, val_y = validation_data
                self.evaluate(val_x, val_y, batch_size=batch_size, verbose=1)

    def evaluate(self, x, y=None, batch_size=32, verbose=1):
        """
        Returns the loss value and metrics for the model in test mode.
        """
        self.eval()
        is_iterator = hasattr(x, '__iter__') and not isinstance(x, (list, tuple))
        
        total_loss = 0.0
        total_metrics = {m: 0.0 for m in self.metrics}
        steps = 0
        
        if is_iterator:
            for batch_x, batch_y in x:
                predictions = self(batch_x)
                loss_val = self.loss_fn(predictions, batch_y)
                val = loss_val.to_list()
                total_loss += val[0] if isinstance(val, list) else val
                
                batch_metrics = self._compute_metrics(predictions, batch_y)
                for m, v in batch_metrics.items():
                    total_metrics[m] += v
                steps += 1
        else:
            predictions = self(x)
            loss_val = self.loss_fn(predictions, y)
            val = loss_val.to_list()
            total_loss = val[0] if isinstance(val, list) else val
            
            batch_metrics = self._compute_metrics(predictions, y)
            for m, v in batch_metrics.items():
                total_metrics[m] += v
            steps = 1
            
        avg_loss = total_loss / steps
        avg_metrics = {m: total_metrics[m] / steps for m in self.metrics}
        
        if verbose:
            metrics_str = f"evaluate loss: {avg_loss:.4f}"
            for m, v in avg_metrics.items():
                metrics_str += f" - {m}: {v:.4f}"
            print(metrics_str)
            
        return avg_loss, avg_metrics

    def predict(self, x, batch_size=32):
        """
        Generates output predictions for the input samples.
        """
        self.eval()
        return self(x)
