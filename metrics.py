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