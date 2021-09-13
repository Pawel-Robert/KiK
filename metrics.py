def trajectory_metrics(trajectory, iteration_num):
    """This function calculates all important statistics, e.g. win rate, episode length, etc."""
    raise NotImplementedError

def batch_metrics(trajectory_batch, iteration_num):
    """This function calculates average metrics"""
    raise NotImplementedError