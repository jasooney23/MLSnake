### MLSnake
___Code last updated 22-07-2023___

A machine learning project that utilizes Deep Q Learning to train an agent to play the "snake" game. Written mainly in Python and TensorFlow. With only a couple hours of training, the newest agent is regularly able to achieve scores similar to or better than my own. 

Features:

* Combines techniques from contemporary research, including Double Deep Q-Networks (DDQNs) and experience replay. These techniques help ensure faster convergence of the model and more efficient + stable learning respectively.
* Uses a CNN at the base model for the agent, as it is particularly suited to computer vision-type tasks (i.e. looking at a game or grid of pixels). Substantially outperforms simple DNNs with just dense layers.
* To fulfill the unique requirements of reinforcement learning and its parameter updates, the gradients are manually calculated with a little help from automatic differentiation (as opposed to just using model.fit() ).

Please note that some of the documentation is OUT OF DATE or rudimentary, as this repository is used personally only (as such there is no need for extensive documentation).

If you would like to test out the code:
1. Clone this repo
2. Install NumPy and TensorFlow (recommended that you install a gpu-compatible version)
3. Run ```python Automatic_Script.py```
4. Watch it learn!

The first little while it will not be very good - it spends this time exploring the environment. It will take some time before it starts to "get" the game, but it's all uphill after that.
