# lcm-benchmarking

Benchmarks for testing PyLCM's split-regime runtime predictions across GPU
hardware and numerical precision.

## Environment

The project uses [Pixi](https://pixi.sh) and supports macOS on Apple silicon and
Linux on x86-64.

```bash
pixi install
```

PyLCM and its temporary unreleased `dags` dependency are installed from their
respective `main` branches on GitHub. The generated `pixi.lock` records the
exact Git revisions used by an environment.

CUDA-specific JAX support will be added alongside the pinned Runpod image.
