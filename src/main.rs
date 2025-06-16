#![recursion_limit = "256"]
mod data;
mod model;
mod training;

use burn::{
    backend::{Autodiff, Cuda},
    optim::AdamConfig,
    tensor::Device,
};
use model::ModelConfig;
use training::{TrainingConfig, train};

fn main() {
    type MyBackend = Cuda<f32, i32>; // Explicit CUDA backend
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    // CUDA device will auto-detect available GPUs
    let device = Device::<MyBackend>::default();

    let artifact_dir = "/tmp/tutorial";
    train::<MyAutoDiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);
    println!("{}", model);
}
