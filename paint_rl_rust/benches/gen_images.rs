use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use paint_rl_rust::TrainingContext;
use pyo3::prelude::*;

fn gen_imgs(num_imgs: usize, training_ctx: &mut TrainingContext) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        training_ctx.gen_imgs(py, num_imgs);
        Ok(())
    })
    .unwrap();
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("gen_imgs 128", move |b| {
        b.iter_batched(
            || 
            {
                TrainingContext::new(
                    64,
                    256,
                    "../temp/all_outputs",
                    "../temp/training/p_net.ptc",
                    "../temp/training/d_net.ptc",
                    50,
                    32,
                    4,
                    Some(100),
                )
            }
            ,
            |mut training_ctx| 
            gen_imgs(black_box(128), &mut training_ctx)
            ,
            BatchSize::LargeInput,
        )
    });
}

criterion_group! {
    name=benches;
    config=Criterion::default().sample_size(10);
    targets=benchmark
}
criterion_main!(benches);
