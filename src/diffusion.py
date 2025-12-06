import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class VE_diffuser():
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = sigma_max ** 2

    def v_t(self, t):
        return t

    def g_t(self, t):
        return jax.grad(self.v_t)(t)
    
    def sample_fwd(self, rng, x0, ts):
        """
        Sample from the forward SDE
        x0 : (B, d)
        ts : (B)
        """
        noise = jax.random.normal(rng, x0.shape)
        v_ts = jnp.expand_dims(self.v_t(ts), axis = tuple(range(1, len(x0.shape))))
        return x0 + jnp.sqrt(v_ts) * noise, noise
    

def log_hat_pt(x, data, t):
    """
    x: single point in R^n or shaped input (e.g. (32,32,1))
    data: array of shape (N, ...) where ... matches x's shape
    t: time (scalar)
    returns: log density log \hat{p}_t(x)
    """
    N = data.shape[0]
    v = t   # assumes v(t) = t

    # Flatten x and data along feature dims
    x_flat = x.reshape(-1)
    data_flat = data.reshape(N, -1)

    # Compute squared distances
    potentials = -jnp.sum((x_flat - data_flat) ** 2, axis=1) / (2 * v)

    # Stable logsumexp
    return logsumexp(potentials, axis=0, b=1/N)


def empirical_score_fn(x, data, t):
    score = jax.grad(lambda x: log_hat_pt(x, data, t))(x)
    return score


def empirical_eps_fn(x, data, t):
    score = jax.grad(lambda x: log_hat_pt(x, data, t))(x)
    sigma = jnp.sqrt(t) # assumes v(t) = t
    eps_pred = -score * sigma
    return eps_pred


def sample_rev(self, rng, eps_fn, num_samples=16, image_shape=(28, 28, 1), num_steps=100, add_last_noise: bool = True):
    # Samples with eps_fn; noise on last step is optional via add_last_noise.

    def step(carry, inp):
        curr_t, is_last = inp
        xs, rng, prev_t = carry
        rng, rng1, rng2 = jax.random.split(rng, 3)

        dt = prev_t - curr_t
        g_t = self.g_t(prev_t)
        v_t = self.v_t(prev_t)
        sigma_t = jnp.sqrt(v_t)

        t = prev_t
        input_t = t

        subkeys = jax.random.split(rng1, num_samples)
        eps_pred = jax.vmap(lambda x, key: eps_fn(x, input_t, key))(xs, subkeys)

        score = -eps_pred / sigma_t
        rev_drift = (g_t ** 2) * score

        noise = jax.random.normal(rng2, xs.shape)
        # If add_last_noise=False => mask out noise on last step
        last_mask = 1.0 if add_last_noise else (1.0 - is_last.astype(xs.dtype))
        noise_scale = jnp.sqrt(dt * g_t) * last_mask

        xs = xs + dt * rev_drift + noise_scale * noise
        return (xs, rng, curr_t), xs

    rng, rng_ = jax.random.split(rng)
    x0 = jnp.sqrt(self.v_t(self.T)) * jax.random.normal(rng_, (num_samples, *image_shape))

    powers = jnp.linspace(0, 1, num_steps)
    ts = self.sigma_min**2 * (self.sigma_max**2 / self.sigma_min**2) ** powers
    ts = jnp.concatenate([jnp.array([0.0]), ts])
    reverse_ts = ts[::-1]

    scan_ts = reverse_ts[1:]  # length = num_steps
    is_last = (jnp.arange(scan_ts.shape[0]) == (scan_ts.shape[0] - 1))
    init = (x0, rng, self.T)
    (xs, _, _), traj = jax.lax.scan(step, init, (scan_ts, is_last))
    return xs, traj
