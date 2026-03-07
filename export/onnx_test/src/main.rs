use ndarray::Array3;
use ort::{session::builder::SessionBuilder, value::Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // ONNX 모델 로드
    let mut session = SessionBuilder::new()?
        .commit_from_file("models/no_mask_proj.onnx")?;

    println!("Model loaded");

    // example input
    let input: Array3<f32> = Array3::zeros((1, 64, 44));

    let input_tensor = Value::from_array(input)?;

    // input name 확인 필요
    let outputs = session.run(vec![
        ("input", input_tensor)
    ])?;

    println!("Outputs: {}", outputs.len());

    Ok(())
}