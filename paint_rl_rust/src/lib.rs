use std::sync::{Arc, RwLock};

use env::SimCanvasEnv;
use image::imageops::FilterType;
use indicatif::ProgressIterator;
use ndarray::Dim;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use tch::Tensor;
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

pub mod env;
pub mod sim_canvas;

#[pyclass]
pub struct TrainingContext {
    env: SimCanvasEnv,
    ref_imgs: Arc<RwLock<Vec<ndarray::Array3<f32>>>>,
    reward_model_path: String,
    p_net_path: String,
}

#[pymethods]
impl TrainingContext {
    #[new]
    pub fn new(
        img_size: u32,
        canvas_size: u32,
        ref_img_path: &str,
        p_net_path: &str,
        reward_model_path: &str,
        max_strokes: u32,
        max_refs: Option<u64>,
    ) -> Self {
        // Load all images
        let mut ref_imgs = Vec::new();
        let path = std::path::Path::new(ref_img_path);
        let count = if let Some(max_refs) = max_refs {
            max_refs
        } else {
            path.read_dir().unwrap().count() as u64
        };
        for dir in path.read_dir().unwrap().take(count as usize).progress_count(count) {
            let dir = dir.unwrap();
            let mut path = dir.path().to_path_buf();
            path.push("final.png");
            let ref_img = image::open(path).unwrap();
            let ref_img = ref_img.resize(img_size, img_size, FilterType::Gaussian);
            let ref_img = ref_img.to_rgb8();
            let ref_img_arr = ndarray::arr1(&ref_img.clone().into_vec())
                .into_shape((img_size as usize, img_size as usize, 3))
                .unwrap()
                .mapv(|x| x as f32)
                .permuted_axes([2, 0, 1])
                / 255.0;
            ref_imgs.push(ref_img_arr.as_standard_layout().into_owned());
        }
        let ref_imgs = Arc::new(RwLock::new(ref_imgs));

        // Set up environment
        let reward_model = Arc::new(RwLock::new(tch::CModule::load(reward_model_path).unwrap()));
        let env_ = SimCanvasEnv::new(
            canvas_size,
            img_size,
            4,
            &ref_imgs,
            max_strokes,
            &reward_model,
        );

        Self {
            ref_imgs,
            env: env_,
            p_net_path: p_net_path.into(),
            reward_model_path: reward_model_path.into(),
        }
    }

    /// Generates a set of images.
    pub fn gen_imgs(
        &mut self,
        py: Python<'_>,
        num_imgs: usize,
        action_scale: f32,
    ) -> Py<PyArray<f32, Dim<[usize; 4]>>> {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let _guard = tch::no_grad_guard();
        let mut rng = rand::thread_rng();
        let ref_imgs = self.ref_imgs.read().unwrap();
        let mut ref_indices: Vec<_> = (0..ref_imgs.len()).collect();
        ref_indices.shuffle(&mut rng);
        let mut img_arr: Vec<ndarray::Array3<f32>> = Vec::with_capacity(num_imgs);
        self.env.set_reward_model(&Arc::new(RwLock::new(
            tch::CModule::load(&self.reward_model_path).unwrap(),
        )));
        let p_net = tch::CModule::load(&self.p_net_path).unwrap();
        for &i in ref_indices[..num_imgs].iter().progress() {
            let ref_img = &ref_imgs[i];
            let ref_img = Tensor::try_from(ref_img).unwrap();
            // Generate image
            let mut obs = self.env.reset_with_index(i).unsqueeze(0);
            let mut done = false;
            let mut trunc = false;
            while !(done || trunc) {
                let tch::IValue::Tuple(results) = p_net
                    .forward_is(&[tch::IValue::Tensor(obs)])
                    .unwrap() else {panic!("Invalid output.")};
                let tch::IValue::Tensor(action_probs_cont) = &results[0] else {panic!("Invalid output.")};
                let tch::IValue::Tensor(action_probs_disc) = &results[1] else {panic!("Invalid output.")};
                let action_cont = action_probs_cont.squeeze()
                    + Tensor::zeros([4], options).normal_(0.0, 1.0) * action_scale as f64;
                let action_disc = sample(action_probs_disc)[0];
                let (obs_, _, done_, trunc_) = self.env.step(&action_cont, action_disc);
                (done, trunc) = (done_, trunc_);
                obs = obs_.unsqueeze(0);
            }
            let data_item: ndarray::ArrayD<f32> =
                Tensor::concatenate(&[self.env.scaled_canvas().unsqueeze(0), ref_img], 0)
                    .as_ref()
                    .try_into()
                    .unwrap();
            img_arr.push(data_item.into_dimensionality().unwrap());
        }
        ndarray::stack(
            ndarray::Axis(0),
            &img_arr.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap()
        .into_pyarray(py)
        .to_owned()
    }
}

/// Returns a list of actions given the probabilities.
fn sample(logits: &Tensor) -> Vec<u32> {
    let num_samples = logits.size()[0];
    let num_weights = logits.size()[1] as usize;
    let mut generated_samples = Vec::with_capacity(num_samples as usize);
    let mut rng = rand::thread_rng();
    let probs = logits.softmax(-1, tch::Kind::Float).clamp(0.0001, 0.9999); // Sampling breaks with really high or low values

    for i in 0..num_samples {
        let mut weights = vec![0.0; num_weights];
        probs.get(i).copy_data(&mut weights, num_weights);
        let builder = WalkerTableBuilder::new(&weights);
        let table = builder.build();
        let action = table.next_rng(&mut rng);
        generated_samples.push(action as u32);
    }

    generated_samples
}

/// Wrapper for environments for vectorization.
pub struct VecEnv {
    pub envs: Vec<SimCanvasEnv>,
    pub num_envs: usize,
}

type VecEnvOutput = (Vec<Tensor>, Vec<f32>, Vec<bool>, Vec<bool>);

impl VecEnv {
    pub fn new(envs: Vec<SimCanvasEnv>) -> Self {
        Self {
            num_envs: envs.len(),
            envs,
        }
    }

    pub fn step(&mut self, actions_cont: &Tensor, actions_disc: &[u32]) -> VecEnvOutput {
        let _guard = tch::no_grad_guard();
        let mut obs_vec = Vec::with_capacity(self.num_envs);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut truncs = Vec::with_capacity(self.num_envs);

        for (i, action_disc) in actions_disc.iter().enumerate() {
            let action_cont = actions_cont.get(i as i64);
            let (obs, reward, done, trunc) = self.envs[i].step(&action_cont, *action_disc);
            rewards.push(reward);
            dones.push(done);
            truncs.push(trunc);

            // Reset if done or truncated
            if done || trunc {
                let obs = self.envs[i].reset();
                obs_vec.push(obs);
            } else {
                obs_vec.push(obs);
            }
        }
        (obs_vec, rewards, dones, truncs)
    }

    pub fn reset(&mut self) -> Vec<Tensor> {
        let _guard = tch::no_grad_guard();
        let mut obs_vec = Vec::with_capacity(self.num_envs);

        for i in 0..self.num_envs {
            let obs = self.envs[i].reset();
            obs_vec.push(obs.copy());
        }
        obs_vec
    }
}

#[pymodule]
fn paint_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TrainingContext>().unwrap();
    Ok(())
}
