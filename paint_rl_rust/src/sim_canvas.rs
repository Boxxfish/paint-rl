use num_traits::{FromPrimitive, ToPrimitive};
use rand::Rng;
use tiny_skia::*;

/// A digital painting simulator.
pub struct SimCanvas {
    pub options: SimCanvasOptions,
    pub last_x: u32,
    pub last_y: u32,
    pub pixmap: Pixmap,
}

impl SimCanvas {
    /// Creates a new SimCanvas instance.
    pub fn new(options: SimCanvasOptions) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            last_x: rng.gen_range(0..options.canvas_size),
            last_y: rng.gen_range(0..options.canvas_size),
            pixmap: Pixmap::new(options.canvas_size, options.canvas_size).unwrap(),
            options,
        }
    }

    /// Clears the canvas and randomizes the start position.
    pub fn clear(&mut self) {
        let color = self.options.clear_color;
        self.pixmap
            .fill(Color::from_rgba(color.0, color.1, color.2, 1.0).unwrap());
        let mut rng = rand::thread_rng();
        self.last_x = rng.gen_range(0..self.options.canvas_size);
        self.last_y = rng.gen_range(0..self.options.canvas_size);
    }

    /// Performs a stroke.
    /// The stroke starts from the last brush position.
    pub fn stroke(
        &mut self,
        brush_index: usize,
        mid_x: i32,
        mid_y: i32,
        end_x: u32,
        end_y: u32,
        brush_params: &[f32]
    ) {
        let brush = &self.options.brushes[brush_index];
        let mut paint = Paint {
            anti_alias: brush.anti_alias,
            ..Default::default()
        };
        let r = brush.color_r.value(brush_params);
        let g = brush.color_r.value(brush_params);
        let b = brush.color_r.value(brush_params);
        paint.set_color(Color::from_rgba(r, g, b, 1.0).unwrap());
        let stroke = Stroke {
            width: brush.brush_diameter.value(brush_params) as f32,
            line_cap: LineCap::Round,
            ..Default::default()
        };
        let path = {
            let mut pb = PathBuilder::new();
            pb.move_to(self.last_x as f32, self.last_y as f32);
            pb.quad_to(mid_x as f32, mid_y as f32, end_x as f32, end_y as f32);
            pb.finish().unwrap()
        };
        self.pixmap
            .stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        self.last_x = end_x;
        self.last_y = end_y;
    }

    /// Moves the cursor to a new position.
    pub fn move_to(&mut self, x: u32, y: u32) {
        self.last_x = x;
        self.last_y = y;
    }

    /// Returns the last brush position.
    pub fn last_brush_pos(&self) -> (u32, u32) {
        (self.last_x, self.last_y)
    }

    /// Returns raw pixels from the canvas.
    pub fn pixels(&self) -> Vec<u8> {
        self.pixmap.data().to_vec()
    }
}

/// Parameters for SimCanvas.
#[derive(Clone)]
pub struct SimCanvasOptions {
    pub canvas_size: u32,
    pub brushes: Vec<BrushOptions>,
    pub clear_color: (f32, f32, f32),
}

impl Default for SimCanvasOptions {
    fn default() -> Self {
        Self {
            canvas_size: 256,
            brushes: vec![BrushOptions::default()],
            clear_color: (1.0, 1.0, 1.0),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ScalingMode {
    Nearest,
    Bilinear,
}

/// Determines if a brush parameter is a constant or range.
#[derive(Copy, Clone)]
pub enum BrushParam<T: ToPrimitive + FromPrimitive + Copy> {
    Constant(T),
    /// When performing a stroke, a value from 0 to 1 can be used.
    /// The arguments are parameter index, min value, and max value.
    Range((usize, T, T)),
}

impl<T: ToPrimitive + FromPrimitive + Copy> BrushParam<T> {
    /// Returns either a constant or a value based on the given param array.
    pub fn value(&self, params: &[f32]) -> T {
        match self {
            BrushParam::Constant(val) => *val,
            BrushParam::Range((index, min, max)) => T::from_f32(
                min.to_f32().unwrap()
                    + (max.to_f32().unwrap() - min.to_f32().unwrap()) * params[*index],
            )
            .unwrap(),
        }
    }
}

#[derive(Copy, Clone)]
/// Options for a brush.
pub struct BrushOptions {
    /// Diameter of the brush in pixels.
    pub brush_diameter: BrushParam<u32>,
    /// The red component of the brush color.
    pub color_r: BrushParam<f32>,
    /// The green component of the brush color.
    pub color_g: BrushParam<f32>,
    /// The blue component of the brush color.
    pub color_b: BrushParam<f32>,
    /// Whether strokes are antialiased.
    pub anti_alias: bool,
}

impl Default for BrushOptions {
    fn default() -> Self {
        Self {
            brush_diameter: BrushParam::Constant(4),
            color_r: BrushParam::Constant(0.0),
            color_g: BrushParam::Constant(0.0),
            color_b: BrushParam::Constant(0.0),
            anti_alias: true,
        }
    }
}
