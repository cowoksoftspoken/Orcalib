use orca_tensor::{Tensor, Backend};
use crate::backend::{Autodiff, AutodiffStorage};
use orca_core::{Result, OrcaError};

/// Extension methods for Autodiff tensors.
pub trait AutogradTensorExt<B: Backend> {
    /// Runs the backward pass from this tensor.
    fn backward(&self) -> Result<()>;

    /// Sets whether this tensor requires gradients.
    fn require_grad(&mut self);

    /// Retrieves the gradient of this tensor.
    fn grad(&self) -> Option<Self> where Self: Sized;

    /// Clears the computational graph and all accumulated gradients.
    fn zero_grad(&self) -> Result<()>;
    
    /// Retrieves the underlying inner primal tensor without the autodiff wrapper.
    fn primal(&self) -> Tensor<B>;
    
    /// Returns a new tensor detached from the current graph (no node_id).
    fn detach(&self) -> Self where Self: Sized;

    /// Overwrites the gradient of this tensor on the tape.
    fn set_grad(&self, grad: &Self) -> Result<()> where Self: Sized;
}

impl<B: Backend> AutogradTensorExt<B> for Tensor<Autodiff<B>> {
    fn backward(&self) -> Result<()> {
        let node_id = self.storage().node_id.ok_or_else(|| {
            OrcaError::InternalError("Cannot call backward on a tensor that doesn't require gradients".into())
        })?;

        let backend = self.backend();
        let inner_backend = backend.inner();
        
        // Seed gradient with 1.0s
        let num_elements = self.shape().num_elements();
        let ones: Vec<f32> = vec![1.0; num_elements];
        let root_grad = inner_backend.from_f32_slice(self.shape(), &ones)?;

        // Lock tape and run backward
        let tape_arc = backend.tape();
        let mut tape = tape_arc.lock().map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        tape.execute_backward(node_id, root_grad, inner_backend)?;
        
        Ok(())
    }

    fn require_grad(&mut self) {
        let backend = self.backend();
        let tape_arc = backend.tape();
        let mut tape = match tape_arc.lock() {
            Ok(t) => t,
            Err(_) => return,
        };
        let id = tape.generate_id();
        self.storage_mut().node_id = Some(id);
    }

    fn grad(&self) -> Option<Self> {
        let node_id = self.storage().node_id?;
        let backend = self.backend();
        let tape_arc = backend.tape();
        let tape = tape_arc.lock().ok()?;
        
        let grad_storage = tape.get_grad(node_id)?;
        
        let autodiff_storage = AutodiffStorage {
            primal: grad_storage.clone(),
            node_id: None,
        };
        
        Some(Tensor::from_raw_parts(
            backend.clone(),
            autodiff_storage,
            self.shape().clone(),
            self.strides().to_vec(),
            self.dtype(),
        ))
    }

    fn zero_grad(&self) -> Result<()> {
        let backend = self.backend();
        let tape_arc = backend.tape();
        let mut tape = tape_arc.lock().map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        
        tape.clear();
        
        Ok(())
    }

    fn primal(&self) -> Tensor<B> {
        Tensor::from_raw_parts(
            self.backend().inner().clone(),
            self.storage().primal.clone(),
            self.shape().clone(),
            self.strides().to_vec(),
            self.dtype(),
        )
    }

    fn detach(&self) -> Self {
        Tensor::from_raw_parts(
            self.backend().clone(),
            AutodiffStorage {
                primal: self.storage().primal.clone(),
                node_id: None,
            },
            self.shape().clone(),
            self.strides().to_vec(),
            self.dtype(),
        )
    }

    fn set_grad(&self, grad: &Self) -> Result<()> {
        let node_id = self.storage().node_id.ok_or_else(|| {
            OrcaError::InternalError("Cannot set gradient on a tensor without node_id".into())
        })?;
        let backend = self.backend();
        let tape_arc = backend.tape();
        let mut tape = tape_arc.lock().map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        tape.set_grad(node_id, grad.storage().primal.clone());
        Ok(())
    }
}
