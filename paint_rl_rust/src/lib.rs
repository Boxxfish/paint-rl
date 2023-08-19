use std::sync::{Arc, RwLock};

use env::SimCanvasEnv;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressIterator};
use ndarray::{Axis, Dim};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rand::{seq::SliceRandom, Rng};
use tch::Tensor;
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

pub mod env;
pub mod sim_canvas;

#[pyclass]
pub struct TrainingContext {
    w_ctxs: Vec<WorkerContext>,
    ref_imgs: Arc<RwLock<Vec<ndarray::Array3<f32>>>>,
    ground_truth_imgs: Arc<RwLock<Vec<ndarray::Array2<f32>>>>,
    reward_model_path: String,
    p_net_path: String,
    num_workers: usize,
    num_steps: usize,
    num_envs: usize,
    canvas_size: u32,
    img_size: u32,
    max_strokes: u32,
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
        num_steps: usize,
        max_refs: Option<u64>,
    ) -> Self {
        // Load all images
        let mut ref_imgs = Vec::new();
        let mut ground_truth_imgs = Vec::new();
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

            // Read reference image
            let mut path = dir.path().to_path_buf();
            path.push("final.png");
            let ref_img = image::open(path).unwrap();
            let ref_img = ref_img.resize(img_size, img_size, FilterType::Gaussian);
            let ref_img = ref_img.to_rgb8();
            let ref_img_arr = ndarray::arr1(&ref_img.clone().into_vec())
                .into_shape((img_size as usize, img_size as usize, 3))
                .unwrap()
                .mapv(f32::from)
                .permuted_axes([2, 0, 1])
                / 255.0;
            ref_imgs.push(ref_img_arr.as_standard_layout().into_owned());

            // Read ground truth image
            let mut path = dir.path().to_path_buf();
            path.push("outlines.png");
            let ground_truth_img = image::open(path).unwrap();
            let ground_truth_img =
                ground_truth_img.resize(img_size, img_size, FilterType::Gaussian);
            let ground_truth_img = ground_truth_img.to_rgb8();
            let ground_truth_img_arr = ndarray::arr1(&ground_truth_img.clone().into_vec())
                .into_shape((img_size as usize, img_size as usize, 3))
                .unwrap()
                .mapv(f32::from)
                .permuted_axes([2, 0, 1])
                .select(Axis(0), &[0])
                .remove_axis(Axis(0))
                / 255.0;
            ground_truth_imgs.push(ground_truth_img_arr.as_standard_layout().into_owned());
        }
        let ref_imgs = Arc::new(RwLock::new(ref_imgs));
        let ground_truth_imgs = Arc::new(RwLock::new(ground_truth_imgs));

        // Set up workers
        let reward_model = Arc::new(RwLock::new(tch::CModule::load(reward_model_path).unwrap()));
        let envs_per_worker = num_envs / num_workers as u32;
        let mut w_ctxs = Vec::new();
        for _ in 0..num_workers {
            let mut env = VecEnv::new(
                (0..(envs_per_worker))
                    .map(|_| {
                        SimCanvasEnv::new(
                            canvas_size,
                            img_size,
                            4,
                            &ref_imgs,
                            &ground_truth_imgs,
                            max_strokes,
                        )
                    })
                    .collect(),
                &reward_model,
            );

            let w_ctx = WorkerContext {
                normalize: NormalizeReward::new(envs_per_worker),
                last_obs: Tensor::stack(&env.reset(), 0),
                env,
                rollout_buffer: RolloutBuffer::new(
                    &[7, 64, 64],
                    &[&[1], &[1], &[1]],
                    &[&[32 * 32], &[32 * 32], &[2]],
                    tch::Kind::Int,
                    envs_per_worker,
                    num_steps as u32,
                ),
            };
            w_ctxs.push(w_ctx);
        }

        Self {
            ref_imgs,
            p_net_path: p_net_path.into(),
            reward_model_path: reward_model_path.into(),
            num_workers,
            w_ctxs,
            num_steps,
            num_envs: num_envs as usize,
            max_strokes,
            canvas_size,
            img_size,
            ground_truth_imgs,
        }
    }

    /// Generates a set of images.
    pub fn gen_imgs(
        &mut self,
        py: Python<'_>,
        num_imgs: usize,
    ) -> Py<PyArray<f32, Dim<[usize; 4]>>> {
        let _guard = tch::no_grad_guard();

        let p_net = Arc::new(RwLock::new(tch::CModule::load(&self.p_net_path).unwrap()));
        let mut img_arr: Vec<ndarray::Array3<f32>> = Vec::with_capacity(num_imgs);
        let reward_model = Arc::new(RwLock::new(
            tch::CModule::load(&self.reward_model_path).unwrap(),
        ));
        let envs_per_worker = self.num_envs / self.num_workers;
        let envs: Vec<_> = (0..self.num_workers)
            .map(|_| {
                VecEnv::new(
                    (0..(envs_per_worker))
                        .map(|_| {
                            SimCanvasEnv::new(
                                self.canvas_size,
                                self.img_size,
                                4,
                                &self.ref_imgs,
                                &self.ground_truth_imgs,
                                self.max_strokes,
                            )
                        })
                        .collect(),
                    &reward_model,
                )
            })
            .collect();
        let mut handles = Vec::new();
        let bar = ProgressBar::new(num_imgs as u64);
        for mut env in envs {
            let p_net = p_net.clone();
            let num_workers = self.num_workers;
            let bar = bar.clone();
            let handle = std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                let mut obs = Tensor::stack(&env.reset(), 0);
                let mut img_arr_thread: Vec<ndarray::Array3<f32>> = Vec::with_capacity(num_imgs);
                let obs_select = Tensor::from_slice(&[4, 5, 6, 2]);
                while img_arr_thread.len() < num_imgs / num_workers {
                    // Choose action
                    let (action_mid, action_end, action_down) =
                        sample_model(&p_net.read().unwrap(), &obs);

                    // Compute stroke action
                    let actions_cont = discs_to_strokes(&action_mid, &action_end);

                    let (obs_, _, done, mut trunc) = env.step(&actions_cont, &action_down);

                    // If any images have finished, add them to our array
                    for (i, (&done, trunc)) in done.iter().zip(&mut trunc).enumerate() {
                        if done || *trunc {
                            let data_item: ndarray::ArrayD<f32> = obs
                                .get(i as i64)
                                .index_select(0, &obs_select)
                                .as_ref()
                                .try_into()
                                .unwrap();
                            img_arr_thread.push(data_item.into_dimensionality().unwrap());
                            bar.inc(1);
                            if img_arr_thread.len() >= num_imgs / num_workers {
                                break;
                            }
                        }
                    }
                    obs = Tensor::stack(&obs_, 0);
                }
                img_arr_thread
            });
            handles.push(handle);
        }

        // Process data once finished
        for handle in handles {
            let mut img_arr_thread = handle.join().unwrap();
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

    /// Performs one iteration of the rollout process.
    /// Returns buffer tensors and entropy.
    pub fn rollout(
        &mut self,
    ) -> (
        PyTensor,
        Vec<PyTensor>,
        Vec<PyTensor>,
        PyTensor,
        PyTensor,
        PyTensor,
    ) {
        let _guard = tch::no_grad_guard();
        let p_net = Arc::new(tch::CModule::load(&self.p_net_path).expect("Couldn't load module."));
        let reward_model = Arc::new(RwLock::new(
            tch::CModule::load(&self.reward_model_path).unwrap(),
        ));

        let num_steps = self.num_steps;
        let mut handles = Vec::new();
        let bar = ProgressBar::new((self.num_workers * num_steps) as u64);
        for mut w_ctx in self.w_ctxs.drain(0..self.w_ctxs.len()) {
            let p_net = p_net.clone();
            let bar = bar.clone();
            w_ctx.env.reward_model = reward_model.clone();
            let handle = std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                let mut obs = w_ctx.last_obs.copy();
                for _ in 0..num_steps {
                    // Choose action
                    let (action_mid_probs, action_end_probs, action_down_probs) =
                        get_act_probs(&p_net, &obs);
                    let action_mid = sample(&action_mid_probs);
                    let action_end = sample(&action_end_probs);
                    let action_down = sample(&action_down_probs);
                    let actions_cont = discs_to_strokes(&action_mid, &action_end);
                    let actions = [&action_mid, &action_end, &action_down];
                    let action_probs = [action_mid_probs, action_end_probs, action_down_probs];

                    let (obs_, rewards, dones, truncs) =
                        w_ctx.env.step(&actions_cont, &action_down);
                    w_ctx.rollout_buffer.insert_step(
                        &obs,
                        &actions
                            .iter()
                            .map(|action| {
                                Tensor::from_slice(
                                    &action.iter().copied().map(i64::from).collect::<Vec<_>>(),
                                )
                                .unsqueeze(1)
                            })
                            .collect::<Vec<_>>(),
                        &action_probs,
                        &Tensor::from_slice(
                            &rewards.iter().copied().map(f64::from).collect::<Vec<_>>(),
                        ),
                        &dones,
                        &truncs,
                    );
                    obs = Tensor::stack(&obs_, 0);
                    bar.inc(1);
                }

                w_ctx.rollout_buffer.insert_final_step(&obs);
                w_ctx.last_obs = obs;

                w_ctx
            });
            handles.push(handle);
        }

        // Process data once finished
        for handle in handles {
            let w_ctx = handle.join().unwrap();
            self.w_ctxs.push(w_ctx);
        }
        bar.finish();

        // Copy the contents of each rollout buffer
        let state_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.states)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let act_buffer = (0..3)
            .map(|i| {
                PyTensor(Tensor::concatenate(
                    &self
                        .w_ctxs
                        .iter()
                        .map(|w_ctx| &w_ctx.rollout_buffer.actions[i])
                        .collect::<Vec<&Tensor>>(),
                    1,
                ))
            })
            .collect();
        let act_probs_buffer = (0..3)
            .map(|i| {
                PyTensor(Tensor::concatenate(
                    &self
                        .w_ctxs
                        .iter()
                        .map(|w_ctx| &w_ctx.rollout_buffer.action_probs[i])
                        .collect::<Vec<&Tensor>>(),
                    1,
                ))
            })
            .collect();
        let reward_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.rewards)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let done_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.dones)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let trunc_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.truncs)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        for w_ctx in self.w_ctxs.iter_mut() {
            w_ctx.rollout_buffer.next = 0;
        }

        (
            PyTensor(state_buffer),
            act_buffer,
            act_probs_buffer,
            PyTensor(reward_buffer),
            PyTensor(done_buffer),
            PyTensor(trunc_buffer),
        )
    }
}

/// State of each worker.
pub struct WorkerContext {
    pub env: VecEnv,
    pub normalize: NormalizeReward,
    pub last_obs: Tensor,
    pub rollout_buffer: RolloutBuffer,
}

/// Returns a list of actions given the probabilities.
fn sample(log_probs: &Tensor) -> Vec<u32> {
    let num_samples = log_probs.size()[0];
    let num_weights = log_probs.size()[1] as usize;
    let mut generated_samples = Vec::with_capacity(num_samples as usize);
    let mut rng = rand::thread_rng();
    let probs = log_probs.exp().clamp(0.0001, 0.9999); // Sampling breaks with really high or low values

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

/// Returns action probabilities from a model.
fn get_act_probs(p_net: &tch::CModule, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
    let tch::IValue::Tuple(results) = p_net
    .forward_is(&[tch::IValue::Tensor(obs.copy())])
    .unwrap() else {panic!("Invalid output.")};
    let tch::IValue::Tensor(action_probs_mid) = &results[0] else {panic!("Invalid output.")};
    let tch::IValue::Tensor(action_probs_end) = &results[1] else {panic!("Invalid output.")};
    let tch::IValue::Tensor(action_probs_down) = &results[2] else {panic!("Invalid output.")};
    (
        action_probs_mid.copy(),
        action_probs_end.copy(),
        action_probs_down.copy(),
    )
}

/// Samples actions from a model.
fn sample_model(p_net: &tch::CModule, obs: &Tensor) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (action_probs_mid, action_probs_end, action_probs_down) = get_act_probs(p_net, obs);
    let action_mid = sample(&action_probs_mid);
    let action_end = sample(&action_probs_end);
    let action_down = sample(&action_probs_down);
    (action_mid, action_end, action_down)
}

/// Converts discrete actions to strokes.
fn discs_to_strokes(action_mid: &[u32], action_end: &[u32]) -> Tensor {
    let mut actions_cont = Vec::new();
    let quant_size = 32; // TODO: Pass this in as an argument
    for (&mid_index, end_index) in action_mid.iter().zip(action_end) {
        let mid_y = mid_index / quant_size;
        let mid_x = mid_index - mid_y * quant_size;
        let end_y = end_index / quant_size;
        let end_x = end_index - end_y * quant_size;
        let action_cont =
            Tensor::from_slice(&[mid_x as f64, mid_y as f64, end_x as f64, end_y as f64])
                / quant_size as f64;
        actions_cont.push(action_cont);
    }
    Tensor::stack(&actions_cont, 0)
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
                // if score >= 0.95 && self.envs[i].counter >= 4 {
                //     *done = true;
                //     let obs = self.envs[i].reset();
                //     obs_vec[i] = obs;
                // }
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

struct RunningMeanStd {
    pub mean: Tensor,
    pub var: Tensor,
    pub count: f32,
}

impl RunningMeanStd {
    fn new() -> Self {
        let mean = Tensor::zeros([], (tch::Kind::Float, tch::Device::Cpu));
        let var = Tensor::ones([], (tch::Kind::Float, tch::Device::Cpu));
        let count = 0.0001;
        Self { mean, var, count }
    }

    fn update(&mut self, x: &Tensor) {
        let batch_mean = Tensor::mean(x, tch::Kind::Float);
        let batch_var = x.var(true);
        let batch_count = x.size()[0];
        self.update_from_moments(batch_mean, batch_var, batch_count);
    }

    fn update_from_moments(&mut self, batch_mean: Tensor, batch_var: Tensor, batch_count: i64) {
        (self.mean, self.var, self.count) = update_mean_var_count_from_moments(
            &self.mean,
            &self.var,
            self.count,
            batch_mean,
            batch_var,
            batch_count,
        );
    }
}

fn update_mean_var_count_from_moments(
    mean: &Tensor,
    var: &Tensor,
    count: f32,
    batch_mean: Tensor,
    batch_var: Tensor,
    batch_count: i64,
) -> (Tensor, Tensor, f32) {
    let delta = &batch_mean.subtract(mean);
    let tot_count = count + batch_count as f32;

    let new_mean = mean + &delta.multiply_scalar(batch_count as f64 / tot_count as f64);
    let m_a = count * var;
    let m_b = batch_var * batch_count;
    let m2 = m_a + m_b + count * (batch_count as f32) / tot_count * delta.square();
    let new_var = m2.divide_scalar(tot_count as f64);
    let new_count = tot_count;

    (new_mean, new_var, new_count)
}

pub struct NormalizeReward {
    return_rms: RunningMeanStd,
    returns: Tensor,
    gamma: f32,
    epsilon: f32,
}
impl NormalizeReward {
    fn new(num_envs: u32) -> Self {
        let return_rms = RunningMeanStd::new();
        let returns = Tensor::zeros([num_envs as i64], (tch::Kind::Float, tch::Device::Cpu));
        let gamma = 0.99;
        let epsilon = 0.0000001;
        Self {
            return_rms,
            returns,
            gamma,
            epsilon,
        }
    }

    fn processes_rews(&mut self, rews: &Tensor, dones: &Tensor) -> Tensor {
        self.returns = (self.gamma * (1 - dones.to_kind(tch::Kind::Float))) * &self.returns + rews;
        self.normalize(rews)
    }

    fn normalize(&mut self, rews: &Tensor) -> Tensor {
        self.return_rms.update(&self.returns);
        rews / Tensor::sqrt(&(self.epsilon + &self.return_rms.var))
    }
}

pub type Size<'a> = &'a [i64];

/// A pared down rollout buffer for collecting transitions.
/// The data should be sent to Python for actual training.
pub struct RolloutBuffer {
    num_envs: u32,
    num_steps: u32,
    next: i64,
    states: Tensor,
    actions: Vec<Tensor>,
    action_probs: Vec<Tensor>,
    rewards: Tensor,
    dones: Tensor,
    truncs: Tensor,
}

impl RolloutBuffer {
    pub fn new(
        state_shape: Size,
        action_shapes: &[Size],
        action_probs_shapes: &[Size],
        action_dtype: tch::Kind,
        num_envs: u32,
        num_steps: u32,
    ) -> Self {
        let k = tch::Kind::Float;
        let d = tch::Device::Cpu;
        let options = (k, d);
        let state_shape = [&[num_steps as i64 + 1, num_envs as i64], state_shape].concat();
        let action_shapes: Vec<_> = action_shapes
            .iter()
            .map(|action_shape| [&[num_steps as i64, num_envs as i64], *action_shape].concat())
            .collect();
        let action_probs_shapes: Vec<_> = action_probs_shapes
            .iter()
            .map(|action_probs_shape| {
                [&[num_steps as i64, num_envs as i64], *action_probs_shape].concat()
            })
            .collect();
        let next = 0;
        let states = Tensor::zeros(state_shape, options).set_requires_grad(false);
        let actions = action_shapes
            .iter()
            .map(|action_shape| {
                Tensor::zeros(action_shape, (action_dtype, d)).set_requires_grad(false)
            })
            .collect();
        let action_probs = action_probs_shapes
            .iter()
            .map(|action_probs_shape| {
                Tensor::zeros(action_probs_shape, options).set_requires_grad(false)
            })
            .collect();
        let rewards =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        let dones =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        let truncs =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        Self {
            num_envs,
            num_steps,
            next,
            states,
            actions,
            action_probs,
            rewards,
            dones,
            truncs,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_step(
        &mut self,
        states: &Tensor,
        actions: &[Tensor],
        action_probs: &[Tensor],
        rewards: &Tensor,
        dones: &[bool],
        truncs: &[bool],
    ) {
        let _guard = tch::no_grad_guard();
        self.states.get(self.next).copy_(states);
        for i in 0..self.actions.len() {
            self.actions[i].get(self.next).copy_(&actions[i]);
            self.action_probs[i].get(self.next).copy_(&action_probs[i]);
        }
        self.rewards.get(self.next).copy_(rewards);
        self.dones.get(self.next).copy_(&Tensor::from_slice(dones));
        self.truncs
            .get(self.next)
            .copy_(&Tensor::from_slice(truncs));

        self.next += 1;
    }

    pub fn insert_final_step(&mut self, state: &Tensor) {
        let _guard = tch::no_grad_guard();
        self.states.get(self.next).copy_(state);
    }
}
