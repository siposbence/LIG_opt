import time

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch


def evaluation_fn(step,speed, power):
    return 300/speed*power*(2-abs(0.1-step))


def easy_objective(config):
    # Hyperparameters
    speed, power, step = config["speed"], config["power"], config["steps"]
    print(speed, power, step)

    intermediate_score = evaluation_fn(step, speed, power)
        # Feed the score back back to Tune.
    tune.report(iterations=1, mean_loss=intermediate_score)
    time.sleep(1)




algo = BayesOptSearch(utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,
        "xi": 0.0
    })
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()
analysis = tune.run(
        easy_objective,
        name="my_exp",
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=20,
        config={
            "steps": tune.uniform(0.05, 0.2), #tune.choice([1,2,3]),
            "speed": tune.uniform(100, 500),
            "power": tune.uniform(100, 600)
        })

print("Best hyperparameters found were: ", analysis.best_config)
