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

The `cuda12` environment installs JAX with CUDA 12 support for the pinned
Runpod image:

```bash
pixi run -e cuda12 python -c 'import jax; print(jax.devices())'
```

The Runpod environment uses template `runpod-torch-v280`, backed by image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
