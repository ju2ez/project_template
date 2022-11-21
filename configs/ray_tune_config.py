from ray import tune

"""
Ray Tune search spaces can be configured here    
"""

def get_example_search_space():
    search_space = {
        "batch_size": tune.grid_search([32]),
        "max_lr": tune.grid_search([3e-4]),
        "base_r": tune.grid_search([0]),
        "num_epochs": tune.grid_search([5]),
        "momentum": tune.grid_search([0.9, 0.8])
    }
    return search_space
