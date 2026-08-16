import jax.numpy as jnp
import lcm

x = (jnp.ones(1, dtype=jnp.float32) + 1).block_until_ready()
device = next(iter(x.devices()))

assert device.platform == "gpu", f"Expected a GPU, got {device}"
print(f"PyLCM imported and JAX ran on {device}.")
