def trajectory_metrics(trajectory, iteration_num):
    """This function calculates all important statistics, e.g. win rate, episode length, etc."""
    # for a in trajectory:
    #    win_rate = sum
    raise NotImplementedError

def batch_metrics(trajectory_batch, iteration_num):
    """This function calculates average metrics"""
    first_wins = 0
    second_wins = 0
    length = 0
    avg_draws = 0
    for trajectory in trajectory_batch:
        # print(trajectory[-1][3])
        first_wins += max(trajectory[-2][3],0)
        second_wins += min(trajectory[-2][3],0)
        length += len(trajectory)
        avg_first_wins = first_wins/len(trajectory_batch)
        avg_second_wins = abs(second_wins) / len(trajectory_batch)
        avg_length = length/len(trajectory_batch)
        avg_draws = round(1 - avg_first_wins - avg_second_wins,3)
    return avg_first_wins, avg_second_wins, avg_draws, avg_length





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