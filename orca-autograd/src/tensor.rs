use orca_tensor::{Tensor, Backend};
use crate::backend::Autodiff;
use orca_core::{Result, OrcaError};

/// Extension methods for Autodiff tensors.
pub trait AutogradTensorExt<B: Backend> {
    /// Runs the backward pass from this tensor.
    fn backward(&self) -> Result<()>;

    /// Sets whether this tensor requires gradients.
    fn require_grad(&mut self);

    /// Retrieves the gradient of this tensor.
    fn grad(&self) -> Option<Tensor<B>>;

    /// Clears the computational graph and all accumulated gradients.
    fn zero_grad(&self) -> Result<()>;
    
    /// Retrieves the underlying inner primal tensor without the autodiff wrapper.
    fn primal(&self) -> Tensor<B>;
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
        let mut tape = tape_arc.lock().unwrap();
        tape.execute_backward(node_id, root_grad, inner_backend);
        
        Ok(())
    }

    fn require_grad(&mut self) {
        let backend = self.backend();
        let tape_arc = backend.tape();
        let mut tape = tape_arc.lock().unwrap();
        let id = tape.generate_id();
        self.storage_mut().node_id = Some(id);
    }

    fn grad(&self) -> Option<Tensor<B>> {
        let node_id = self.storage().node_id?;
        let backend = self.backend();
        let tape_arc = backend.tape();
        let tape = tape_arc.lock().unwrap();
        
        let grad_storage = tape.get_grad(node_id)?;
        
        Some(Tensor::from_raw_parts(
            backend.inner().clone(),
            grad_storage.clone(),
            self.shape().clone(),
            self.strides().to_vec(),
            self.dtype(),
        ))
    }

    fn zero_grad(&self) -> Result<()> {
        let backend = self.backend();
        let tape_arc = backend.tape();
        let mut tape = tape_arc.lock().unwrap();
        
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
}
