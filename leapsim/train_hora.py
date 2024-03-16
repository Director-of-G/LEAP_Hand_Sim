# --------------------------------------------------------
# This file train_hora.py is modified from train.py from
# the Hora repository.
# Origin:
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import isaacgym

import os, shutil
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from leapsim.tasks import isaacgym_task_map
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
from leapsim.utils.utils import set_np_formatting, set_seed, git_hash, git_diff_config

from hora.algo.ppo.ppo import PPO


## OmegaConf & Hydra Config

@hydra.main(config_name='config', config_path='./cfg')
def main(config: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config.default_run_name}_{time_str}"

    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    cfg_dict = omegaconf_to_dict(config)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    config.seed = set_seed(config.seed)
    import pdb; pdb.set_trace()

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = isaacgym_task_map[config.task_name](
        cfg=cfg_dict["task"],
        rl_device=config.rl_device,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    # dump config dict
    experiment_dir = os.path.join('runs', run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copyfile("cfg/task/LeapHandRot.yaml", os.path.join(experiment_dir, "LeapHandRot.yaml"))
    shutil.copyfile("cfg/train/LeapHandRotPPO.yaml", os.path.join(experiment_dir, "LeapHandRotPPO.yaml"))

    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))

    assert config.train.params.config.ppo
    agent = eval("PPO")(env, experiment_dir, full_config=config)
    if config.test:
        agent.restore_test(config.train.params.load_path)
        agent.test()
    else:
        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            experiment_dir, "nn"
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f'are you intentionally going to overwrite files in {best_ckpt_path}, type yes to continue \n')
            if user_input != 'yes':
                exit()

        agent.restore_train(config.train.params.load_path)
        agent.train()


if __name__ == '__main__':
    main()
