#![recursion_limit = "256"]
mod data;
mod model;
mod training;
use burn::{
    backend::{Autodiff, Metal},
    optim::AdamConfig,
    tensor::Device,
};
use model::ModelConfig;
use training::{TrainingConfig, train};
fn main() {
    type MyBackend = Metal<f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    let device = Device::<MyBackend>::default();
    let artifact_dir = "/tmp/tutorial";
    train::<MyAutoDiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);
    println!("{}", model)
}
