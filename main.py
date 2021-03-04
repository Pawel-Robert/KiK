import tensorflow as tf

import numpy as np

height = 10
width = 10
player = 1

#tworzymy środowisko gry
env = KiKEnv(height, width, player)
n_actions = env.height * env.width

#funkcja tworząca modele o odpowiedniej architekturze
def create_kik_model():
  model = tf.keras.models.Sequential([
      # First Dense layer
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=n_actions, activation=None)
  ])
  return model


#funckja wybierająca akcję za pomocą modelu
def choose_action(model, observation):
    # add batch dimension to the observation if only a single example was provided
    # observation = np.expand_dims(observation, axis=0) if single else observation

    logits = model.predict(observation)

    #losujemy akcję, która jest dozwolonym ruchem
    while True:
        action = tf.random.categorical(logits, num_samples=1)
        if env.is_allowed_move(action):
            break
    
    action = action.numpy().flatten()

    return action

#klasa obiektów, które zapamiętują ruchy w postaci trzech list: obserwacji, akcji oraz nagród
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


# Helper function to combine a list of Memory objects into a single Memory.
#     This will be very useful for batching.
def aggregate_memories(memories):
    batch_memory = Memory()

    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)

    return batch_memory


# Instantiate a single Memory buffer
memory = Memory()

# funkcja normalizująca zmienną x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)

# funkcja obliczjąca funckję straty
def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)  # TODO
    return loss

# funckja trenująca model na podstawie wybranej akcji oraz obserwacji
def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      logits = model(observations)

      loss = compute_loss(logits, actions, discounted_rewards)
  grads = tape.gradient(loss, model.trainable_variables) 
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  
  # Learning rate and optimizer
  learning_rate = 1e-3
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  # instantiate kik agent
  kik_model = create_kik_model()

  # to track our progress
  smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)

  if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
  for i_episode in range(500):

      plotter.plot(smoothed_reward.get())

      # Restart the environment
      observation = env.reset()
      memory.clear()

      while True:
          # using our observation, choose an action and take it in the environment
          action = choose_action(kik_model, observation)
          next_observation, reward, done, info = env.step(action)
          # add to memory
          memory.add_to_memory(observation, action, reward)

          # is the episode over? did you crash or do so well that you're done?
          if done:
              # determine total reward and keep a record of this
              total_reward = sum(memory.rewards)
              smoothed_reward.append(total_reward)

              # initiate training - remember we don't know anything about how the
              #   agent is doing until it has crashed!
              train_step(kik_model, optimizer,
                         observations=np.vstack(memory.observations),
                         actions=np.array(memory.actions),
                         discounted_rewards=discount_rewards(memory.rewards))

              # reset the memory
              memory.clear()
              break
          # update our observatons
          observation = next_observation


### kik training! ###

# Learning rate and optimizer
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# instantiate kik agent
kik_model = create_kik_model()

# to track our progress
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
for i_episode in range(500):

    plotter.plot(smoothed_reward.get())

    # Restart the environment
    observation = env.reset()
    memory.clear()

    while True:
        # using our observation, choose an action and take it in the environment
        action = choose_action(kik_model, observation)
        next_observation, reward, done, info = env.step(action)
        # add to memory
        memory.add_to_memory(observation, action, reward)

        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            smoothed_reward.append(total_reward)

            # initiate training - remember we don't know anything about how the
            #   agent is doing until it has crashed!
            train_step(kik_model, optimizer,
                       observations=np.vstack(memory.observations),
                       actions=np.array(memory.actions),
                       discounted_rewards=discount_rewards(memory.rewards))

            # reset the memory
            memory.clear()
            break
        # update our observatons
        observation = next_observation