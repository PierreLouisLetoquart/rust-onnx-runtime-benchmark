use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::Array4;
use ort::{GraphOptimizationLevel, Session};

fn main() -> Result<(), Box<dyn Error>> {
    let image_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("imgs");

    // Load the model
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("model")
                .join("mobilenetv2-12.onnx"),
        )?;

    println!("{model:?}");

    // Process each image in the directory
    let mut total_time = 0.0;
    let mut image_count = 0;

    for entry in std::fs::read_dir(&image_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            image_count += 1;

            // Load and preprocess image
            let image = preprocess_image(&path);

            // Run inference and time it
            let start_time = Instant::now();

            // Run the backbone model
            let outputs = model.run(ort::inputs!["input" => image.clone()]?)?;
            let output = outputs["output"].try_extract_tensor::<f32>()?;

            let elapsed_time = start_time.elapsed().as_secs_f32();
            total_time += elapsed_time;

            // Log the inference time and results
            println!(
                "Image {:?}: Inference time = {:.2} seconds",
                path, elapsed_time
            );

            println!("Predictions: {:?}", output);
        }
    }

    // Log overall statistics
    if image_count > 0 {
        println!(
            "\nProcessed {} images with average inference time = {:.2} seconds",
            image_count,
            total_time / image_count as f32
        );
    } else {
        println!("No images found in the specified directory.");
    }

    Ok(())
}

fn preprocess_image(image_path: &Path) -> Array4<f32> {
    let image = image::open(image_path).unwrap().to_rgb8();

    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);

    Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
}
