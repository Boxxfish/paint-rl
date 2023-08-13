use std::sync::{Arc, RwLock};

use env::SimCanvasEnv;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressIterator};
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
    w_ctxs: Vec<WorkerContext>,
    ref_imgs: Arc<RwLock<Vec<ndarray::Array3<f32>>>>,
    reward_model_path: String,
    p_net_path: String,
    num_workers: usize,
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
        num_envs: u32,
        num_workers: usize,
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
        for dir in path
            .read_dir()
            .unwrap()
            .take(count as usize)
            .progress_count(count)
        {
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

        // Set up workers
        let reward_model = Arc::new(RwLock::new(tch::CModule::load(reward_model_path).unwrap()));
        let w_ctxs = (0..num_workers)
            .map(|_| WorkerContext {
                env: VecEnv::new(
                    (0..(num_envs / num_workers as u32))
                        .map(|_| {
                            SimCanvasEnv::new(canvas_size, img_size, 4, &ref_imgs, max_strokes)
                        })
                        .collect(),
                    &reward_model,
                ),
            })
            .collect();

        Self {
            ref_imgs,
            p_net_path: p_net_path.into(),
            reward_model_path: reward_model_path.into(),
            num_workers,
            w_ctxs,
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

        let p_net = Arc::new(RwLock::new(tch::CModule::load(&self.p_net_path).unwrap()));
        let mut img_arr: Vec<ndarray::Array3<f32>> = Vec::with_capacity(num_imgs);
        let reward_model = Arc::new(RwLock::new(
            tch::CModule::load(&self.reward_model_path).unwrap(),
        ));
        for w_ctx in &mut self.w_ctxs {
            w_ctx.env.reward_model = reward_model.clone();
        }
        let mut handles = Vec::new();
        let bar = ProgressBar::new(num_imgs as u64);
        for mut w_ctx in self.w_ctxs.drain(0..self.w_ctxs.len()) {
            let p_net = p_net.clone();
            let num_workers = self.num_workers;
            let bar = bar.clone();
            let handle = std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                let mut obs = Tensor::stack(&w_ctx.env.reset(), 0);
                let mut img_arr_thread: Vec<ndarray::Array3<f32>> = Vec::with_capacity(num_imgs);
                let obs_select = Tensor::from_slice(&[4, 5, 6, 2]);
                while img_arr_thread.len() < num_imgs / num_workers {
                    // Choose player action
                    let tch::IValue::Tuple(results) = p_net.read().unwrap()
                    .forward_is(&[tch::IValue::Tensor(obs.copy())])
                    .unwrap() else {panic!("Invalid output.")};
                    let tch::IValue::Tensor(action_probs_cont) = &results[0] else {panic!("Invalid output.")};
                    let tch::IValue::Tensor(action_probs_disc) = &results[1] else {panic!("Invalid output.")};
                    let action_cont = action_probs_cont
                        + Tensor::zeros([w_ctx.env.num_envs as i64, 4], options).normal_(0.0, 1.0)
                            * action_scale as f64;
                    let action_disc = sample(action_probs_disc);

                    let (obs_, _, done, trunc) = w_ctx.env.step(&action_cont, &action_disc);

                    // If any images have finished, add them to our array
                    for (i, (&done, trunc)) in done.iter().zip(trunc).enumerate() {
                        if done || trunc {
                            let data_item: ndarray::ArrayD<f32> = obs
                                .get(i as i64)
                                .index_select(0, &obs_select)
                                .as_ref()
                                .try_into()
                                .unwrap();
                            img_arr_thread.push(data_item.into_dimensionality().unwrap());
                            bar.inc(1);
                        }
                    }
                    obs = Tensor::stack(&obs_, 0);
                }
                (w_ctx, img_arr_thread)
            });
            handles.push(handle);
        }

        // Process data once finished
        for handle in handles {
            let (w_ctx, mut img_arr_thread) = handle.join().unwrap();
            self.w_ctxs.push(w_ctx);
            img_arr.append(&mut img_arr_thread);
        }
        bar.finish();

        // Stack images
        ndarray::stack(
            ndarray::Axis(0),
            &img_arr.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap()
        .into_pyarray(py)
        .to_owned()
    }
}

/// State of each worker.
pub struct WorkerContext {
    pub env: VecEnv,
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
    pub reward_model: Arc<RwLock<tch::CModule>>,
}

type VecEnvOutput = (Vec<Tensor>, Vec<f32>, Vec<bool>, Vec<bool>);

impl VecEnv {
    pub fn new(envs: Vec<SimCanvasEnv>, reward_model: &Arc<RwLock<tch::CModule>>) -> Self {
        Self {
            num_envs: envs.len(),
            envs,
            reward_model: reward_model.clone(),
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

        // Compute last scores
        let mut reward_inpts = Vec::with_capacity(self.num_envs);
        for i in 0..self.num_envs {
            let scaled_canvas = self.envs[i].scaled_canvas();
            let reward_inpt = self.envs[i].reward_input(Some(&scaled_canvas));
            reward_inpts.push(reward_inpt);
        }
        let reward_inpts = Tensor::concatenate(&reward_inpts, 0);

        let mut scores_buf: Vec<f32> = vec![0.0; self.num_envs];
        let scores = self
            .reward_model
            .read()
            .unwrap()
            .forward_ts(&[reward_inpts])
            .unwrap();
        scores.copy_data(&mut scores_buf, self.num_envs);

        for (i, (done, trunc)) in dones.iter_mut().zip(&truncs).enumerate() {
            if !(*done || *trunc) {
                let score = scores_buf[i];
                rewards[i] += score - self.envs[i].last_score;
                self.envs[i].last_score = score;

                // If reward model is very certain, mark as done
                if score >= 0.95 && self.envs[i].counter >= 4 {
                    *done = true;
                    let obs = self.envs[i].reset();
                    obs_vec[i] = obs;
                }
            }
        }

        // Compute last scores of reset environments
        let mut reward_inpts = Vec::with_capacity(self.num_envs);
        for i in 0..self.num_envs {
            let reward_inpt = self.envs[i].reward_input(None);
            reward_inpts.push(reward_inpt);
        }
        let reward_inpts = Tensor::concatenate(&reward_inpts, 0);

        let mut scores_buf: Vec<f32> = vec![0.0; self.num_envs];
        let scores = self
            .reward_model
            .read()
            .unwrap()
            .forward_ts(&[reward_inpts])
            .unwrap();
        scores.copy_data(&mut scores_buf, self.num_envs);

        for (i, (&done, &trunc)) in dones.iter().zip(&truncs).enumerate() {
            if done || trunc {
                self.envs[i].last_score = scores_buf[i];
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
