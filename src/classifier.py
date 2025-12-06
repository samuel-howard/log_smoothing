import jax
import jax.numpy as jnp

from functools import partial
from typing import Any, Collection, Sequence

import flax.linen as nn
import optax

# define classifier model
class CNN(nn.Module):
    # optionally output the penultimate layer for use in FID
    @nn.compact
    def __call__(self, x, return_features=False):
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))

        x = nn.Conv(64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        if return_features:
            return x  # shape (B, 128)
        
        x = nn.Dense(10)(x)
        return x
    

class Classifier:
    
    def __init__(self, model):
        self.model = model

    # define training functions
    def classifier_loss(self, params, rng, xs, labels):
        logits = self.model.apply(params, xs)  # (B, 10)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, (accuracy,)

    def classifier_update(self, params, rng, opt_state, xs, labels):
        (loss, (accuracy,)), grads = jax.value_and_grad(self.classifier_loss, has_aux=True)(params, rng, xs, labels)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, accuracy

    @partial(jax.jit, static_argnums=(0, 6))
    def classifier_train_step(self, rng, params, opt_state, train_images, train_labels, batch_size):
        
        batch_rng, update_rng, rng = jax.random.split(rng, 3)
        idx = jax.random.randint(batch_rng, (batch_size,), 0, train_images.shape[0])
        xs = train_images[idx]
        labels = train_labels[idx]

        params, opt_state, loss, accuracy = self.classifier_update(params, update_rng, opt_state, xs, labels)

        return params, opt_state, loss, accuracy
    
    def train(self, rng, classifier_params, train_images, train_labels, valid_images, valid_labels, train_config):
        learning_rate = train_config['learning_rate']
        batch_size = train_config['batch_size']
        num_steps = train_config['num_steps']
        num_checkpoints = train_config['num_checkpoints']

        self.optimizer = optax.adam(learning_rate)
        opt_state = self.optimizer.init(classifier_params)

        checkpoint_freq = num_steps // num_checkpoints

        losses = []
        params_lst = []

        # training loop
        for step in range(num_steps):
            rng, rng_ = jax.random.split(rng)
            classifier_params, opt_state, loss, accuracy = self.classifier_train_step(rng_, classifier_params, opt_state, train_images, train_labels, batch_size)
            losses.append(loss)
            if step % 10 == 0:
                valid_logits = self.model.apply(classifier_params, valid_images)
                valid_accuracy = jnp.mean(jnp.argmax(valid_logits, axis=-1) == valid_labels)
                print(f'Step {step}: loss {loss}, train accuracy {accuracy}, valid_accuracy {valid_accuracy}')

            if step % checkpoint_freq == 0:
                params_lst.append(classifier_params)

        return classifier_params, losses, params_lst