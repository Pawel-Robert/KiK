import tensorflow as tf

import numpy as np

height = 10
width = 10

#tworzymy środowisko gry
env = KiKEnv(height, width)
n_actions = env.height * env.width

#funkcja tworząca sieć neuronową o odpowiedniej architekturze
# TODO: dostosować parametry sieci konwolucyjnej do rozmiarów planszy
# TODO: na wejściu jest np.array o rozmiarach height X width - czy to dobrze będzie działać?
def create_kik_model():
  model = tf.keras.models.Sequential([
          # na początku tworzymy kilka warstw konwolucyjnych przetwarzających planszę
          tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
          tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
          # spłaszczamy sieć
          tf.keras.layers.Flatten(),
          # warstwa z maksymalną ilością połączeń
          tf.keras.layers.Dense(128, activation=tf.nn.relu),
          # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy
          tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)
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


# Funkcja pomocnicza agregująca pamięć
#     This will be very useful for batching.
# TODO: tego chyba nigdzie nie używamy
def aggregate_memories(memories):
    batch_memory = Memory()
    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)
    return batch_memory




# funkcja normalizująca zmienną x
# TODO: zrozumieć o co tu chodzi
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
# TODO: to chyba nie ma sensu, bo nagroda jest tylko na końcu rozgrywki
def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

# funkcja obliczjąca funckję straty
# TODO: zrozumieć jak tu działa funkcja straty
def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

# funckja trenująca model na podstawie wybranej akcji oraz obserwacji
# TODO: gdzie należy zaimplementować algorytm przeszukiwania drzewa i w jaki sposób?
def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      logits = model(observations)
      loss = compute_loss(logits, actions, discounted_rewards)
  grads = tape.gradient(loss, model.trainable_variables) 
  optimizer.apply_gradients(zip(grads, model.trainable_variables))




### Trenujemy model grający w KiK ###

# ustalamy parametry uczenia: tempo oraz optymalizator
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# tworzymy instację modelu agenta
kik_model = create_kik_model()

# Tworzymy instancję klasy pamięci gracza 1
memory = Memory()

# aby śledzić postęp gry -> funkcje z pakietu MIT TODO: trzeba to ogranąć
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # co to znaczy? (clear if exists)
for i_episode in range(500):

    plotter.plot(smoothed_reward.get())

    # Restart the environment
    observation = env.reset()
    memory.clear()

    while True:
        # zmieniamy gracza wykonującego ruch (inwolucja graczy)
        player = - player
        # using our observation, choose an action and take it in the environment
        raw_action = choose_action(kik_model, observation)
        # zamieniamy "surową akcję" na współrzędne pola
        action[0] = raw_action // env.width # bierzemy podłogę z dzielenia
        action[1] = raw_action % env.width # reszta z dzielenia
        action[2] = player
        # posługując się naszą akcją wykonujemy krok w środowisku i zbieramy z niego informacje
        next_observation, reward, done, info = env.step(action)
        # aktualizujemy pamięć dla gracza nr 1
        if player = 1:
            memory.add_to_memory(observation, raw_action, reward) # do pamięci dodajemy tylko raw_action, a nie 3-elementową listę
        if done:
            # TODO: nie ma potrzeby niczego sumować, nagroda jest tylko za wygraną....
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