use burn::{
    backend::Autodiff,
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::AdamConfig,
    prelude::*,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: usize,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

use crate::{
    data::MnistBatch,
    model::{Model, ModelConfig},
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artefacts to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should have been saved successfully")
}
