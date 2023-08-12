use std::sync::{Arc, RwLock};

use image::{imageops::FilterType, RgbImage, RgbaImage};
use rand::Rng;
use tch::Tensor;

use crate::sim_canvas::{BrushOptions, BrushParam, SimCanvas, SimCanvasOptions};

/// Environment wrapper for SimCanvas.
pub struct SimCanvasEnv {
    sim_canvas: SimCanvas,
    scaled_size: u32,
    counter: u32,
    ref_imgs: Arc<RwLock<Vec<ndarray::Array3<f32>>>>,
    ref_img: Tensor,
    pub ref_img_index: usize,
    max_strokes: u32,
    last_pen_down: bool,
    prev_frame: Tensor,
    last_score: f32,
    reward_model: Arc<RwLock<tch::CModule>>,
}

impl SimCanvasEnv {
    /// Creates a new environment.
    pub fn new(
        canvas_size: u32,
        scaled_size: u32,
        brush_diameter: u32,
        ref_imgs: &Arc<RwLock<Vec<ndarray::Array3<f32>>>>,
        max_strokes: u32,
        reward_model: &Arc<RwLock<tch::CModule>>,
    ) -> Self {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let canvas_options = SimCanvasOptions {
            canvas_size,
            brushes: vec![BrushOptions {
                brush_diameter: BrushParam::Constant(brush_diameter),
                color_r: BrushParam::Constant(1.0),
                color_g: BrushParam::Constant(1.0),
                color_b: BrushParam::Constant(1.0),
                anti_alias: false,
            }],
            clear_color: (0.0, 0.0, 0.0),
        };
        Self {
            sim_canvas: SimCanvas::new(canvas_options),
            scaled_size,
            counter: 0,
            ref_imgs: ref_imgs.clone(),
            ref_img: Tensor::zeros([], (tch::Kind::Float, tch::Device::Cpu)),
            max_strokes,
            last_pen_down: false,
            prev_frame: Tensor::zeros([2, scaled_size as i64, scaled_size as i64], options),
            last_score: 0.0,
            reward_model: reward_model.clone(),
            ref_img_index: 0,
        }
    }

    /// Performs one step of the simulation.
    /// Returns observation, reward, done flag, and truncation flag.
    pub fn step(&mut self, cont_action: &Tensor, disc_action: u32) -> (Tensor, f32, bool, bool) {
        let _guard = tch::no_grad_guard();
        let canvas_size = self.sim_canvas.options.canvas_size;
        let cont_action = Tensor::clip(
            &(cont_action.squeeze() * canvas_size as f64),
            0.0,
            (canvas_size - 1) as f64,
        );
        let mid_point = (
            cont_action.double_value(&[0]) as i32,
            cont_action.double_value(&[1]) as i32,
        );
        let end_point = (
            cont_action.double_value(&[2]) as u32,
            cont_action.double_value(&[3]) as u32,
        );
        let pen_down = disc_action == 1;
        if pen_down {
            self.sim_canvas
                .stroke(0, mid_point.0, mid_point.1, end_point.0, end_point.1, &[]);
        } else {
            self.sim_canvas.move_to(end_point.0, end_point.1);
        }
        self.counter += 1;
        let trunc = self.counter == self.max_strokes;

        let mut reward = 0.0;

        // Penalize refusing to put down strokes
        if !self.last_pen_down && !pen_down {
            reward = -1.0;
        }
        self.last_pen_down = pen_down;

        // Compute score
        let scaled_canvas = self.scaled_canvas();
        let reward_inpt =
            Tensor::concatenate(&[&self.ref_img, &scaled_canvas.unsqueeze(0)], 0)
                .unsqueeze(0)
                .to_kind(tch::Kind::Float);
        let score = self
            .reward_model
            .read()
            .unwrap()
            .forward_ts(&[reward_inpt])
            .unwrap()
            .double_value(&[]) as f32;
        reward += score - self.last_score;
        self.last_score = score;

        // If reward model is very certain, mark as done
        let mut done = false;
        if score >= 0.95 && self.counter >= 4 {
            done = true;
        }

        let scale_ratio = self.scaled_size as f32 / self.sim_canvas.options.canvas_size as f32;
        let pos_channel = self.gen_pos_channel(
            (self.sim_canvas.last_x as f32 * scale_ratio) as u32,
            (self.sim_canvas.last_y as f32 * scale_ratio) as u32,
            self.scaled_size,
        );
        let this_frame = Tensor::stack(&[scaled_canvas, pos_channel], 0);
        let obs = Tensor::concatenate(&[&self.prev_frame, &this_frame, &self.ref_img], 0);
        self.prev_frame = this_frame;

        (obs, reward, done, trunc)
    }

    /// Generates the positional channel.
    pub fn gen_pos_channel(&self, x: u32, y: u32, img_size: u32) -> Tensor {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let pos_layer_x =
            Tensor::arange_start((0 - x as i64), (img_size as i64 - x as i64), options)
                .unsqueeze(0)
                .repeat([img_size as i64, 1])
                / img_size as f64;
        let pos_layer_y =
            Tensor::arange_start((0 - y as i64), (img_size as i64 - y as i64), options)
                .unsqueeze(0)
                .repeat([img_size as i64, 1])
                .t_()
                / img_size as f64;
        Tensor::sqrt(&(&pos_layer_x * &pos_layer_x + &pos_layer_y * &pos_layer_y))
    }

    /// Returns a scaled version of the canvas.
    pub fn scaled_canvas(&self) -> Tensor {
        let canvas_size = self.sim_canvas.options.canvas_size;
        let canvas_img =
            RgbaImage::from_raw(canvas_size, canvas_size, self.sim_canvas.pixels()).unwrap();
        let canvas_img = image::imageops::resize(
            &canvas_img,
            self.scaled_size,
            self.scaled_size,
            FilterType::Gaussian,
        );
        Tensor::from_slice(canvas_img.into_vec().as_slice())
            .reshape([self.scaled_size as i64, self.scaled_size as i64, 4])
            .permute([2, 0, 1])
            .get(0)
            .to_kind(tch::Kind::Float)
            / 255.0
    }

    /// Resets the environment.
    pub fn reset(&mut self) -> Tensor {
        let mut rng = rand::thread_rng();
        let ref_imgs_len = self.ref_imgs.read().unwrap().len();
        let index = rng.gen_range(0..ref_imgs_len);
        self.reset_with_index(index)
    }

    /// Resets the environment with an index.
    pub fn reset_with_index(&mut self, index: usize) -> Tensor {
        self.ref_img_index = index;
        let _guard = tch::no_grad_guard();
        let options = (tch::Kind::Float, tch::Device::Cpu);
        self.counter = 0;
        let ref_imgs = &self.ref_imgs.read().unwrap();
        self.ref_img = Tensor::try_from(&ref_imgs[index]).unwrap();
        self.last_pen_down = false;
        let scale_ratio = self.scaled_size as f32 / self.sim_canvas.options.canvas_size as f32;
        let pos_channel = self.gen_pos_channel(
            (self.sim_canvas.last_x as f32 * scale_ratio) as u32,
            (self.sim_canvas.last_y as f32 * scale_ratio) as u32,
            self.scaled_size,
        );
        self.prev_frame = Tensor::zeros(
            [2, self.scaled_size as i64, self.scaled_size as i64],
            options,
        );
        let this_frame = Tensor::stack(&[self.scaled_canvas(), pos_channel], 0);
        let obs = Tensor::concatenate(&[&self.prev_frame, &this_frame, &self.ref_img], 0);
        self.prev_frame = this_frame;
        self.sim_canvas.clear();
        
        // Compute score
        let reward_inpt = Tensor::concatenate(
            &[
                &self.ref_img,
                &Tensor::zeros(
                    [1, self.scaled_size as i64, self.scaled_size as i64],
                    options,
                ),
            ],
            0,
        )
        .unsqueeze(0)
        .to_kind(tch::Kind::Float);
    
        self.last_score = self
            .reward_model
            .read()
            .unwrap()
            .forward_ts(&[reward_inpt])
            .unwrap()
            .double_value(&[]) as f32;

        obs
    }

    /// Sets the current reward model.
    pub fn set_reward_model(&mut self, reward_model: &Arc<RwLock<tch::CModule>>) {
        self.reward_model = reward_model.clone();
    }
}
