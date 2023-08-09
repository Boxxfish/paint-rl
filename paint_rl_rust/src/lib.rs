use pyo3::prelude::*;

pub mod sim_canvas;

#[pymodule]
fn paint_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    Ok(())
}
