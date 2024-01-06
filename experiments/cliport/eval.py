import os
import pickle
import json
import time

import numpy as np

from environments.ur5_simulation.env import UR5SimulationEnv


from agents import Agent, agent_factory
from executor import SimpleExecutor
from utils import logging

def run_exp(
    agent_name: str,
    task_name: str,
    seed: int,
    save_folder: str,
    n_repeats: int,
    save_logs: list = []
):
    for log_types in save_logs:
        assert log_types in ["result_json", "html", "pkl", "record"]

    """Run a single task."""

    all_results = {}
    name = '{}-{}'.format(task_name, agent_name)

    # Save path for results.
    json_name = f"results.json"
    print(f"Save path for results: {save_folder}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_json = os.path.join(save_folder, f'{name}-{json_name}')

    results = []
    mean_reward = 0.0

    # Run testing for each training run.
    for repeat_idx in range(n_repeats):
        total_reward = 0

        np.random.seed(seed)

        # Create a new logger 
        # NOTE: May overwrite previous logs. Make sure they are saved before calling this
        logger = logging.CustomLogger()
        logging.set_logger(logger)

        # Initialize environment.
        exp_env = UR5SimulationEnv()
        sim_env = exp_env._ur5_sim_env
        # Setup the environment
        info = exp_env.setup(task_name)
        task = info["task"]
        lang_goal = '\n'.join(task.lang_goals)
        plan_img = exp_env.get_image()
        print(f'Lang Goal: {lang_goal}')

        # Initialize agent.
        agent = agent_factory(agent_name)

        reward = 0

        # Start recording video (NOTE: super slow)
        if 'record' in log_types:
            video_name = f'{repeat_idx+1:06d}'
            sim_env.start_rec(video_name)
        
        # plan_result = agent.try_plan(prompt=lang_goal, image=plan_img)
        # if not plan_result.success:
        #     print("Planning failed. Details: \n", plan_result)
        # print("Planning succeed. Details: \n", plan_result)

        # # Plan execution
        # executor = SimpleExecutor(exp_env.get_execution_context())
        # # Build a context containing the masks for the plan code to access
        # context = {"regions": [mask["segmentation"] for mask in plan_result.masks]}
        # executor.execute_plan(plan_result.plan_code, additional_context=context)
        time.sleep(10)

        # Get reward
        reward, info = task.reward()
        done = task.done()

        total_reward += reward
        print(f'Reward: {reward:.3f}')

        info = {"agent": agent_name,
                "task": task_name,
                "seed": seed,
                }
        results.append((total_reward, info))
        mean_reward = np.mean([r for r, i in results])
        print(f'Mean: {mean_reward} | Task: {task_name} | Agent: {agent_name}')

        # End recording video
        if 'record' in log_types:
            sim_env.end_rec()

        if 'pkl' in log_types:
            logger.save_logs(os.path.join(save_folder, f"logs_{repeat_idx + 1}.pkl"))

        if 'html' in log_types:
            logger.save_logs(os.path.join(save_folder, f"html_logs_{repeat_idx + 1}.html"))

        print(mean_reward)
        

    # Save results in a json file.
    if "result_json" in save_logs:
        with open(save_json, 'w') as f:
            json.dump(all_results, f, indent=4)


def main():
    agent_name = "SegVLM"
    task_name = "put-block-in-bowl-seen-colors"

    task_name = "cliport_" + task_name
    seed = 114514
    save_folder = f"results/{agent_name}/{task_name}/{seed}"

    run_exp(agent_name, task_name, seed, save_folder, n_repeats=1, save_logs=["result_json", "html", "pkl"])


[

 'packing-boxes-pairs-seen-colors', #?
 'packing-boxes-pairs-unseen-colors',
 'packing-boxes-pairs-full',
 'packing-seen-google-objects-seq',
 'packing-unseen-google-objects-seq',
 'packing-seen-google-objects-group',
 'packing-unseen-google-objects-group',
 'put-block-in-bowl-seen-colors',
 'put-block-in-bowl-unseen-colors',
 'put-block-in-bowl-full',
 'stack-block-pyramid-seq-seen-colors',
 'stack-block-pyramid-seq-unseen-colors',
 'stack-block-pyramid-seq-full',
 'separating-piles-seen-colors',
 'separating-piles-unseen-colors',
 'separating-piles-full',
 'towers-of-hanoi-seq-seen-colors',
 'towers-of-hanoi-seq-unseen-colors',
 'towers-of-hanoi-seq-full']

if __name__ == '__main__':
    main()