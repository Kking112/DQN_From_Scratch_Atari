## Overview

This Repo contains an implementation of the DQN algorithm for the Atari 2600 games, based on the paper <b>"Playing Atari with Deep Reinforcement Learning"</b> by DeepMind.<b> Although I used Claude to create the guide (Guide.md), <u> I wrote all of the actual code myself.</u></b> This is simply a learning exercise; I'm not trying to create the optimial implementation of DQN by any means, and this implementation is definitely not optimized. Feedback and constructive criticism is not only welcomed, but encouraged.

This repo is also in progress, so some of the code may be messy and unorganized.

## Guide

The guide is a detailed walkthrough of the code, and is located in the Guide.md file. It is a work in progress, and will be updated as the code is developed.

## Code

The code is located in the single_file_DQN_Atari.py file. It is a work in progress, and will be updated as the code is developed. It currently only supports CUDA, however I plan to add support for CPU & MPS in the future. You can find the dependencies in pyproject.toml.

Tested on Ubuntu 24.04, not tested on any other OS. To install dependencies, I recommend using uv. You must have an existing logs directory within the working directory, although you can change the parameters to output logs elsewhere.

Note that we are using the Adam optimizer, which is not the optimizer used in the paper. This is simply because Adam is more stable and faster to train. You may choose to use RMSprop or another optimizer if you perfer, but I would recommend sticking with Adam.

## TODO

- Add MPS/CPU Support
- Add extensive Eval scripts w/ visualization of training and testing
- Add video recording
- Test and add support for all Atari envs
- Seperate single file into multiple modules to be imported into a main file
- Experiment with different hyperparameters