import tensorflow as tf
import numpy as np
from network import build_network

N_PARTICLES = 2
N_DATA = int(1e5)
BATCH_SIZE = 256

masses = np.random.uniform(0.1, 10., (N_PARTICLES))
# get a target scaling
targets = np.random.uniform(10., 100., (1, 1, N_PARTICLES, 1))


def random_four_vectors(n_data=256):
    momenta = np.random.normal(0., 1., (n_data//BATCH_SIZE, BATCH_SIZE, N_PARTICLES, 3))
    energies = np.sqrt(np.sum(momenta ** 2, axis=-1) + np.sum(masses**2)) 
    return np.concatenate((np.expand_dims(energies, axis=-1), momenta), axis=-1)


network = build_network(n_particles=N_PARTICLES, batch_size=BATCH_SIZE)

# Some information about the network
print(network.summary())
print("-"*20)
print("Trainable Variables:")
for layer in network.layers:
    print("-"*20)
    print("{}:".format(layer.name))
    print("Trainable Variables:")
    print(layer.trainable_variables)


data  = random_four_vectors(n_data = N_DATA)
noise = np.random.normal(0., 1., (N_DATA//BATCH_SIZE, BATCH_SIZE, N_PARTICLES, 4))
target_vectors = data * targets

# Feed some values into the network
output = network.predict([data, noise])
output = np.reshape(output, (-1, 2, 4))
print("Values:")
for i_particle , (m_p, s_p) in enumerate(zip(np.mean(output, axis=0), np.std(output, axis=0))):
    print("Particle #{}".format(i_particle))
    for m, s in zip(m_p, s_p):
        print("\t{0:2.2f}+-{1:2.2f}".format(m, s))

# Train the network onto the targets
network.compile(optimizer='adam',
        loss='mse',
        )

print("Target values:")
for n_p, target in enumerate(np.squeeze(targets)):
    print("{0:2d}: {1:2.2f}".format(n_p, target))

network.fit(x=[data, noise], y=[target_vectors], epochs=200)

# Feed again some values into the network
output = network.predict([data, noise])
output = np.reshape(output, (-1, 2, 4))
print("Values:")
for i_particle , (m_p, s_p) in enumerate(zip(np.mean(output, axis=0), np.std(output, axis=0))):
    print("Particle #{}".format(i_particle))
    for m, s in zip(m_p, s_p):
        print("\t{0:2.2f}+-{1:2.2f}".format(m, s))

print("New smearing variables:")
print(network.layers[-1].trainable_variables[0])
print(network.layers[-1].trainable_variables[1])
