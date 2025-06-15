mod data;
mod model;
mod training;
use crate::model::ModelConfig;
use burn::{backend::Metal, tensor::Device};
fn main() {
    type MyBackend = Metal<f32, i32>;

    let device = Device::<MyBackend>::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);
    println!("{}", model)
}
