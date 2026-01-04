import jax
import jax.numpy as jnp


def get_smoothed_fn(func, sigma, n=1000):
    """
    Return a smoothed version of `func` by convolving with a Gaussian kernel.
    """
    @jax.jit
    def smoothed_func(x, t, key):
        zs = jax.random.normal(key, shape=(n, *x.shape))
        # Generate the offsets for the mesh grid
        offsets = sigma * zs
        # Shift the point x by each offset in the meshgrid
        shifted_points = x + offsets
        # Evaluate func at each of these shifted points
        # vals, potentials = jax.vmap(lambda offset: func(offset, t, key))(shifted_points)
        vals = jax.vmap(lambda offset: func(offset, t, key))(shifted_points)
        # Return the mean of these values (key not used in func here)
        return jnp.mean(vals, axis=0)
    
    return smoothed_func



def get_proj_idx_interpolant(sample, true_manifold_alphas, true_manifold_samples):
    '''
    sample: current position
    alphas: scalar params for the true manifold points along the 1d curve
    true_manifold_samples: true manifold approximation
    '''
    dists = jnp.sum((sample - true_manifold_samples)**2, axis=(1, 2, 3))
    idx = jnp.argmin(dists)
    alpha = true_manifold_alphas[idx]
    return alpha, true_manifold_samples[idx]

def get_manifold_translated_smoothed_fn(func, sigma, true_manifold_alphas, true_manifold_samples, n=1000):
    """
    Return a smoothed version of `func` by smoothing along the translated manifold
    """
    @jax.jit
    def smoothed_func(x, t, key):
        zs = jax.random.normal(key, shape=(n, *x.shape))
        zs = sigma * zs

        _, proj_manifold = get_proj_idx_interpolant(x, true_manifold_alphas, true_manifold_samples)
        _, new_proj_manifolds = jax.vmap(get_proj_idx_interpolant, in_axes=(0, None, None))(proj_manifold + zs, true_manifold_alphas, true_manifold_samples)

        shifted_points = x + (new_proj_manifolds - proj_manifold)

        # # Evaluate func at each of these shifted points
        vals = jax.vmap(lambda offset: func(offset, t, key))(shifted_points)
        # Return the mean of these values (key not used in func here)
        return jnp.mean(vals, axis=0)

    return smoothed_func