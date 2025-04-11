# Causal Policy Explanation for Autonomous Driving in IsaacSim

Install IsaacSim and IsaacLab from [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

``NOTE: Prefer using .venv for the installation.``

Verify IsaacSim installation by running:

```bash
isaacsim
```

Verify IsaacLab installation by running:

```bash
./IsaacLab/isaaclab.bat -p scripts/tutorials/00_sim/create_empty.py
```

## Environment Info

- Windows 11, RTX 4070
- Python 3.10.11
- IsaacSim 4.5.0
- IsaacLab 2.0.2
- CUDA 12.1

# Project Goals

### **Novelty:** This project attempts to train a causal explanation model that directly links the environmental inputs to the actions of a simulated autonomous driving vehicle.

### **Goal:** To train a causal explanation model (DAG) using the perceived experience of a trained reinforcement learning model.

## Inputs

* **Detected Road Features**  
  * In order to simplify the project, and to create a larger focus on the development of a causal explanation model, we map inputs directly to the presence of certain road features in front of the simulated car. This serves as a sufficient analog for the detection of these features in the real world without having to use computer vision directly. These features could be detected by using a custom-made module that queries the environment?  
  * This gives us a set of categorical inputs detected by a perception sensor/module on the simulated car.  
    * stop\_sign\_distance: Float (0 if \>= 1000 units away) or Boolean  
    * left\_turn\_lane\_distance: Float (0 if \>= 1000 units away) or Boolean  
    * right\_turn\_lane\_distance: Float (0 if \>= 1000 units away) or Boolean  
    * and more…?  
* **IsaacSim Sensor Data:**  
  * Definitely  
    * Linear velocity (x and y components in the body frame).  
    * Angular velocity (z component in the world frame).  
    * Current coordinates?  
  * If using waypoints  
    * The coordinates of the target waypoint.  
    * Position error relative to the current waypoint.  
    * Heading error relative to the current waypoint.  
  * Potentially   
    * Previous throttle and steering actions **for temporal context**

## Outputs

* **Trained RL Policy:** DNN that takes the above inputs and outputs the robot's throttle and steering actions to navigate.  
* **Causal Explanation Model**  
  * Given a specific state, outputs  
    * **Causal Influence Scores:** Quantified scores indicating the estimated causal influence of each input feature and sensor reading on the robot's current throttle and steering actions. For example:  
      * "The presence of a stop sign had a \-0.9 causal influence on the throttle (deceleration)."  
      * "The 'turn lane left' feature had a \+0.7 causal influence on the steering (increased left turn)."  
      * "The heading error had a \+0.5 causal influence on the throttle (increased speed to correct)."  
    * Potentially, **Probabilistic Causal Graph**  
      * A DAG that shows the probabilistic causal relationships between the input features/sensors and the output actions. Edge weights would represent the probability or strength of the causal link

## Training the RL Model

* **Environment:** Randomized road generation with a guaranteed number of features (e.g. at least 2 stop signs, etc) and waypoints for simplified navigation  
* **Observation Space**: Detected road features \+ IsaacSim Sensor Data  
* **Reward Function**:  
  * Punish and reset on touch wall/sidewalk  
  * Reward on efficient reaching of waypoints?

## Training the Causal Explanation Model

* **Data Collection:** During the RL training process, collect a dataset of state transitions. For each step, record the detected road features, the sensor data, the current waypoint, and the throttle and steering actions taken by the agent.  
* **Causal Inference:** Use a method suitable for potentially mixed data types (Boolean, Categorical, Numerical) including  
  * **Bayesian Networks with Structure Learning:** Algorithms like the Hill-Climbing or Tabu Search can learn the structure of a Bayesian Network from the collected data, representing probabilistic dependencies (which can be interpreted as causal). (pgmpy?)  
  * Causal forest for Non-linear effects? – scikit-learn?  
  * This means having a separate statistical model or  even shallow NN trained to predict causal influence scores based on state?  
* **Output:** Estimation of the causal impact of each input variable on the agent's actions. This could involve:  
  * **Predicting Actions with Feature Ablation:** Train the model to predict actions. Then, evaluate how the predictions change when specific features are "ablated" (set to a neutral or absent value). The magnitude of the change can indicate the causal influence.  
  * For Bayesian Networks, the learned network structure and conditional probabilities directly represent the relationships?

## Evaluating Data Collected

XRL conventions hope to measure the following features:

| Feature | Description |
| :---: | :---- |
| Faithfulness | Do the measured causal influences make intuitive sense? e.g. the presence of a stop sign should lead to deceleration. |
| Completeness | Does the causal model consider all road features and sensor data? |
| Comprehensibility | Does the output of the model make intuitive and meaningful sense to humans? |
| Robustness | Compare learned DAGs before and after minor perturbations to the agent and environment; was the graph structure stable? |
