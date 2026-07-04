use std::collections::HashMap;
use std::fmt::Debug;
use orca_tensor::Backend;

/// A unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Storage for gradients computed during the backward pass.
#[derive(Debug)]
pub struct Gradients<B: Backend> {
    grads: HashMap<NodeId, B::Storage>,
}

impl<B: Backend> Default for Gradients<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Gradients<B> {
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Retrieve the gradient for a given node.
    pub fn get(&self, id: NodeId) -> Option<&B::Storage> {
        self.grads.get(&id)
    }

    /// Insert or accumulate a gradient for a given node.
    pub fn accumulate(&mut self, id: NodeId, grad: B::Storage, backend: &B) {
        if let Some(existing) = self.grads.get(&id) {
            let sum = backend.accumulate_grad(existing, &grad).unwrap();
            self.grads.insert(id, sum);
        } else {
            self.grads.insert(id, grad);
        }
    }

    pub fn consume(self) -> HashMap<NodeId, B::Storage> {
        self.grads
    }
}

/// A trait for operations that can compute their own backward pass.
pub trait BackwardOp<B: Backend>: Send + Sync + Debug {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B);
}

/// The Wengert List (Tape) that records operations.
#[derive(Debug)]
pub struct Tape<B: Backend> {
    nodes: Vec<Box<dyn BackwardOp<B>>>,
    next_id: usize,
    grads: Gradients<B>,
}

impl<B: Backend> Default for Tape<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Tape<B> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
            grads: Gradients::new(),
        }
    }

    pub fn push_node(&mut self, op: Box<dyn BackwardOp<B>>) {
        self.nodes.push(op);
    }

    pub fn clear(&mut self) {
        self.grads.grads.clear();
        self.nodes.clear();
    }

    pub fn generate_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        NodeId(id)
    }

    pub fn execute_backward(&mut self, root_id: NodeId, root_grad: B::Storage, backend: &B) {
        self.grads.grads.insert(root_id, root_grad);
        for op in self.nodes.iter().rev() {
            op.backward(&mut self.grads, backend);
        }
    }

    pub fn get_grad(&self, id: NodeId) -> Option<&B::Storage> {
        self.grads.get(id)
    }
}
