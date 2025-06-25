"""Quadratic Function
# Flags: doc-Runnable

An example of applying SMAC to optimize a quadratic function.

We use the black-box facade because it is designed for black-box function optimization.
The black-box facade uses a [Gaussian Process][GP] as its surrogate model.
The facade works best on a numerical hyperparameter configuration space and should not
be applied to problems with large evaluation budgets (up to 1000 evaluations).
"""

import numpy as np
import random
import math
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from matplotlib import pyplot as plt
import os
import subprocess
import sys
import json
import pandas as pd
import csv

from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac import BlackBoxFacade
from smac.facade import AbstractFacade
from smac import RunHistory, Scenario
from smac.runhistory.dataclasses import TrialValue
from smac.utils import configspace

import gmail_monitor

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class Simulation:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        rim_radius = Float("rim radius", (0.1, 0.5), default=.25)
        width = Float("width", (0.1, 0.5), default=.25)
        grouser_number = Float("grouser number", (10, 30), default=20)
        grouser_thickness = Float("grouser thickness", (0.001, 0.025), default=.01)
        grouser_height = Float("grouser height", (0.01, 0.15), default=.1)
        control_point_deviation = Float("control point deviation", (-.3, 0.3), default=0)
        wave_number = Integer("wave number", (0, 3), default=1)
        wave_amplitude = Float("wave amplitude", (0, 0.03), default=0.01)


        cs.add([rim_radius, width, grouser_number, grouser_thickness, grouser_height, control_point_deviation, 
                wave_number, wave_amplitude])
        return cs

    #bigger slip ratio is better
    def evaluate(self, config: Configuration, seed: int = 0) -> float:
        slip_ratio = 0
        slip_ratio += config["rim radius"] * 0.6
        slip_ratio += config["width"] * 0.4
        slip_ratio += math.cos(config["grouser number"] * 2*math.pi / 20) * -0.1
        slip_ratio += config["grouser thickness"] * -1
        slip_ratio += config["grouser height"] * 2
        slip_ratio += config["control point deviation"] * 0.1
        slip_ratio += config["wave number"] * -.01
        slip_ratio += config["wave amplitude"] * -1.6
        noise_scale = (random.random()*.4) +.8
        slip_ratio *= noise_scale
        return -slip_ratio

def plot_runs(paramter : str, configs, incumbents, runhistory):  
        for i, config in enumerate(configs):
            if config in incumbents:      
                continue

            label = None
            if i == 0:
                label = "Configuration"
            
            x = config[paramter]   
            f1 = runhistory.get_cost(config)    
            plt.scatter(x, f1, c="blue", alpha=0.1, marker="o", zorder=3000, label=label)

        for i, config in enumerate(incumbents):    
            label = None    
            if i == 0:      
                label = "Incumbent"

            x = config[paramter]    
            f1 = runhistory.get_cost(config)    
            plt.scatter(x, f1, c="red", alpha=1, marker="x", zorder=3000, label=label)

        plt.xlabel(paramter)  
        plt.ylabel("score")  
        plt.title("sim eval")  
        plt.legend()
        plt.show()


def plot_from_smac(smac: HPOFacade, model:Simulation, runhistory) -> None:
    plt.figure()
    configs = smac.runhistory.get_configs()
    incumbents = smac.intensifier.get_incumbents()

    # for wheel_param in list(scenario.configspace.values()):
    #     plot_runs(wheel_param.name, configs, incumbents, runhistory)
    plot_runs("rim radius", configs, incumbents, runhistory)


if __name__ == "__main__":
    model = Simulation()
    n_trials = 4

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials = n_trials)

    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        model.evaluate,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    # incumbent = smac.optimize()
    for trial_num in range(n_trials):
        info = smac.ask()
        assert info.seed is not None

        wheel_param = {
            "rim radius": info.config["rim radius"],
            "width": info.config["width"],
            "grouser number": info.config["grouser number"],
            "grouser thickness": info.config["grouser thickness"],
            "grouser height": info.config["grouser height"],
            "control point deviation": info.config["control point deviation"],
            "wave number": info.config["wave number"],
            "wave amplitude": info.config["wave amplitude"]
        }
        with open("wheel_jsons/wheel_parameters.json", "w") as json_file:
            json.dump(wheel_param, json_file, indent=4)

        simulation_json = {
            "terrain_filepath": "/jet/home/matthies/moonranger_mobility/terrain/GRC_3e5_Reduced_Footprint.csv",
            "wheel_filepath": "/jet/home/matthies/moonranger_mobility/meshes/wheel_"+str(trial_num)+"/wheel.obj",
            "wheel_filepath": "/jet/home/matthies/moonranger_mobility/meshes/wheel_"+str(trial_num)+"/wheel_parameters.json",
            "data_drivepath": "/ocean/projects/mch240013p/matthies/"
        }

        print("start bash script")
        script_result = subprocess.run(["./automated_pipeline.sh", "upload", str(trial_num)], capture_output=True, text=True)
        JOB_ID = 0
        if script_result.returncode == 0:
            output_lines = script_result.stdout.strip().split('\n')
            for line in output_lines:
                if "job id:" in line:
                    JOB_ID = line.split(":")[1]
            if JOB_ID == 0:
                print("job id not found")
            else:
                print("Script success and jobid captured "+ JOB_ID)
        else:
            print("Script failed or filename not capture")
        print("end bash script")
        print("stdout:")
        print(script_result.stdout)
        print("stderr:")
        print(script_result.stderr)

        print("start gmail monitor")
        gmail_monitor.monitor(JOB_ID)
        
        df = pd.read_csv('~/Documents/wheel_sim_data/automation_test_' + JOB_ID + '/output_directory/output.csv')


        avg_dc = model.evaluate(info.config)
        # avg_dc = (df['f_x'] / df['f_z']).mean()

        

        cost = avg_dc
        value = TrialValue(cost=cost)

        smac.tell(info, value)
    
    incumbent = smac.intensifier.get_incumbent()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's plot it too
    plot_from_smac(smac, model, smac.runhistory)
    print(incumbent)

    