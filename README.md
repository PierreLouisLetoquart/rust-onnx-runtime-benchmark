# Benchmark onnx runtime libs in Rust

> This is a super bad and naive bbenchmark, nothing super clever (at least for now lol)

This is a super simple and naive benchmark to compare [ort](https://ort.pyke.io/) and [tract](https://github.com/sonos/tract) for their onnx runtime rust lib.

## Use it

```bash
cargo run -p ort_benchmark
cargo run -p tract_benchmark
```

## Note

There is no post processing for now, no proper logging too (e.g. using tracing lib lol) etc etc.

