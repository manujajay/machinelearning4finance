import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Generate some synthetic "real" financial market return data (replace with real data)
np.random.seed(0)
real_data = np.random.normal(0, 1, (1000, 1))

# Generator Model
input_noise = Input(shape=(10,))
hidden_layer_g = Dense(30, activation='relu')(input_noise)
generated_data = Dense(1, activation='linear')(hidden_layer_g)
generator = Model(inputs=input_noise, outputs=generated_data)

# Discriminator Model
input_real_data = Input(shape=(1,))
hidden_layer_d = Dense(30, activation='relu')(input_real_data)
validity = Dense(1, activation='sigmoid')(hidden_layer_d)
discriminator = Model(inputs=input_real_data, outputs=validity)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# GAN Model
discriminator.trainable = False
gan_output = discriminator(generator(input_noise))
gan = Model(inputs=input_noise, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Training parameters
epochs = 1000
batch_size = 32

# Train GAN
for epoch in range(epochs):
    # Train Discriminator
    noise = np.random.normal(0, 1, (batch_size, 10))
    generated_data = generator.predict(noise)
    real_data_batch = real_data[np.random.randint(0, real_data.shape[0], batch_size)]
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_data_batch, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_data, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 10))
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)

# Generate data to visualize
noise = np.random.normal(0, 1, (1000, 10))
generated_data = generator.predict(noise)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(real_data, alpha=0.5, label='Real Data')
ax.hist(generated_data, alpha=0.5, label='Generated Data')
ax.set_title('GAN for Simulating Financial Market Conditions')
ax.set_xlabel('Market Returns')
ax.set_ylabel('Frequency')
ax.legend()

# Model Description and Key Stats
description = '''Model Description:
The GAN model consists of a Generator and a Discriminator.
The Generator tries to produce synthetic financial data, while the Discriminator tries to distinguish between real and synthetic data.
After training, we use the Generator to simulate different market conditions for assessing potential risks associated with various investment strategies.'''

stats = f'''Key Stats:
- Number of epochs: {epochs}
- Batch size: {batch_size}
- Discriminator Loss: {d_loss[0]:.4f}
- Generator Loss: {g_loss:.4f}'''

fig.text(0.2, 0.55, description, fontsize=6, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
fig.text(0.65, 0.25, stats, fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

# Save plot
plt.savefig('GAN_Financial_Simulation.png')
plt.show()
