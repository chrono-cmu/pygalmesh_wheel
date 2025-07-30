import numpy as np
from ax.api.client import Client, IMetric
from ax.api.configs import RangeParameterConfig
import math
import random
from IPython.core.display import HTML
from matplotlib import pyplot as plt
import random
import json
import subprocess
import gmail_monitor
import pandas as pd
import os
import time
import sys
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
import trimesh

batch_size = 5
num_batches = 2

def evaluate(parameters):
        rim_radius = parameters["rim_radius"]
        width = parameters["width"]
        grouser_number = parameters["grouser_number"]
        grouser_thickness = parameters["grouser_thickness"]
        grouser_height = parameters["grouser_height"]
        control_point_deviation = parameters["control_point_deviation"]
        wave_number = parameters["wave_number"]
        wave_amplitude = parameters["wave_amplitude"]
        score = 0
        score += rim_radius * 0.6
        score += width * 0.4
        score += math.cos(grouser_number * 2*math.pi / 20) * -0.1
        score += grouser_thickness * -1
        score += grouser_height * 2
        score += control_point_deviation * 0.1
        score += wave_number * -.01
        score += wave_amplitude * -1.6
        noise_scale = random.random()*.4 +.8
        score *= noise_scale
        return score

client = Client()
parameters = [
    RangeParameterConfig(
        name="rim_radius", parameter_type="float", bounds=(0.085, 0.110)
    ),
    RangeParameterConfig(
        name="width", parameter_type="float", bounds=(0.04, 0.135)
    ),
    RangeParameterConfig(
        name="grouser_number", parameter_type="int", bounds=(5, 30)
    ),
    
        RangeParameterConfig(
        name="grouser_height", parameter_type="float", bounds=(0.01, 0.04)
    ),
    RangeParameterConfig(
        name="control_point_deviation", parameter_type="float", bounds=(0, 0.3)
    )
    # RangeParameterConfig(
    #     name="grouser_thickness", parameter_type="float", bounds=(0.001, 0.025)
    # ),
    #     RangeParameterConfig(
    #     name="wave_number", parameter_type="int", bounds=(0, 3)
    # ),
    # RangeParameterConfig(
    #     name="wave_amplitude", parameter_type="float", bounds=(0, 0.003)
    # )
]
client.configure_experiment(
    parameters=parameters,
    parameter_constraints=["rim_radius + grouser_height <= 0.115"],
)
metric_name = "score" # this name is used during the optimization loop in Step 5
objective = f"{metric_name}" # minimization is specified by the negative sign
client.configure_optimization(objective=objective)

def construct_generation_strategy(
    generator_spec: GeneratorSpec, node_name: str,
) -> GenerationStrategy:
    """Constructs a Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        model_specs=[generator_spec],
    )
    sobol_node = GenerationNode(
        node_name="Sobol",
        model_specs=[
            GeneratorSpec(
                model_enum=Generators.SOBOL,
                # Let's use model_kwargs to set the random seed.
                model_kwargs={"seed": 0},
            ),
        ],
        transition_criteria=[
            # Transition to BoTorch node once there are 5 trials on the experiment.
            MinTrials(
                threshold=batch_size,
                transition_to=botorch_node.node_name,
                use_all_trials_in_exp=True,
            )
        ]
    )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    return GenerationStrategy(
        name=f"Sobol+{node_name}",
        nodes=[sobol_node, botorch_node]
    )

# Let's construct the simplest version with all defaults.
generation_strategy = construct_generation_strategy(
    generator_spec=GeneratorSpec(model_enum=Generators.BOTORCH_MODULAR),
    node_name="Modular BoTorch",
)

#custom generation strategy
client.set_generation_strategy(
    generation_strategy=generation_strategy,
)

# previous_data = pd.read_csv("./data.csv")
drawbar_job_ids = []
steering_job_ids = []

for batch_number in range(num_batches):
    
    trials = client.get_next_trials(max_trials=batch_size)
    #get batch parameters
    job_array = []
    for trial_index, parameters in trials.items():
        job_array.append([trial_index, parameters, 0, 0])
    #launch every job
    for i in range(len(job_array)):
        trial_index, parameters, _, _ = job_array[i]
        wheel_param = {
            "rim_radius": parameters["rim_radius"],
            "width": parameters["width"],
            "grouser_number": parameters["grouser_number"],
            "grouser_height": parameters["grouser_height"],
            "control_point_deviation": parameters["control_point_deviation"],
            "outer_radius": parameters["rim_radius"] + parameters["grouser_height"]
            # "wave_number": parameters["wave_number"],
            # "wave_amplitude": parameters["wave_amplitude"],
            # "grouser_thickness": parameters["grouser_thickness"],
        }
        # download wheel json
        with open("wheel_jsons/wheel_parameters.json", "w") as json_file:
            json.dump(wheel_param, json_file, indent=4)

        #create wheel
        subprocess.run(["python3", "gen_wheel.py," "-w," "wheel_jsons/wheel_parameters.json"])

        #find mass
        mesh = trimesh.load("wheel.obj")
        volume = mesh.volume
        mass = volume * 2.7 * 1e6
        wheel_param["mass"] = mass
        with open("wheel_jsons/wheel_parameters.json", "w") as json_file:
            json.dump(wheel_param, json_file, indent=4)

        # create json w/ job parameters
        drawbar_job_json = {
            "terrain_filepath": "/jet/home/matthies/moonranger_mobility/terrain/GRC_3e5_Reduced_Footprint",
            "wheel_folder_path": "/jet/home/matthies/moonranger_mobility/meshes/wheel_"+str(trial_index)+"/",
            "data_drivepath": "/ocean/projects/mch240013p/matthies/",
            "sim_endtime": 5,
            "output_dir": "",
            "rotational_velocity": 0.2,
            "step_size": 1e-6,
            "scale_factor": 10,
            "wheel_angle": 0
        }
        # download job json
        with open("job_json/drawbar_job_json.json", "w") as json_file:
            json.dump(drawbar_job_json, json_file, indent=4)

        # create json w/ job parameters
        steering_job_json = {
            "terrain_filepath": "/jet/home/matthies/moonranger_mobility/terrain/GRC_3e5_Reduced_Footprint",
            "wheel_folder_path": "/jet/home/matthies/moonranger_mobility/meshes/wheel_"+str(trial_index)+"/",
            "data_drivepath": "/ocean/projects/mch240013p/matthies/",
            "sim_endtime": 5,
            "output_dir": "",
            "rotational_velocity": 0.2,
            "step_size": 1e-6,
            "scale_factor": 10,
            "wheel_angle": 10
        }
        # download job json
        with open("job_json/steering_job_json.json", "w") as json_file:
            json.dump(steering_job_json, json_file, indent=4)

        # previous_jobs = ['33401656', '33401659', '33401662', "33412051", "33412057"]
        # if trial_index < len(previous_data):
        #     JOB_ID = previous_jobs[trial_index]
    
        # bash script creates wheel mesh, uploads mesh & json, and launches script
        print("start bash script")
        script_result = subprocess.run(["./automated_pipeline.sh", "upload", str(trial_index)], capture_output=True, text=True)
        
        STEERING_JOB_ID = 0
        DRAWBAR_JOB_ID = 0
        if script_result.returncode == 0:
            output_lines = script_result.stdout.strip().split('\n')
            for line in output_lines:
                if "drawbar job id:" in line:
                    DRAWBAR_JOB_ID = line.split(":")[1]
                elif "steering job id:" in line:
                    STEERING_JOB_ID = line.split(":")[1]
            if STEERING_JOB_ID == 0 or DRAWBAR_JOB_ID == 0:
                print("job id not found")
            else:
                print("Script success and jobid captured. Drawbar: "+ DRAWBAR_JOB_ID, " Steering: ", STEERING_JOB_ID)
        else:
            print("Script failed or filename not capture")
        print("end bash script")
        print("stdout:")
        print(script_result.stdout)
        print("stderr:")
        print(script_result.stderr)

        # job_array[i] = [trial_index, updated_parameters, JOB_ID]
        job_array[i] = [trial_index, parameters, DRAWBAR_JOB_ID, STEERING_JOB_ID]
        drawbar_job_ids.append(DRAWBAR_JOB_ID)
        steering_job_ids.append(STEERING_JOB_ID)

    print("start gmail monitor")
    drawbar_job_ids = [row[2] for row in job_array]
    steering_job_ids =  [row[3] for row in job_array]
    all_job_ids = drawbar_job_ids + steering_job_ids
    gmail_monitor.monitor(all_job_ids)

    # slip values currently tested
    slip_values = [-0.1, 0.0, 0.1]
    slip_values_str = ["-0.100000", "0.000000", "0.100000"]
    for trial_index, parameters, drawbar_job_id, steering_job_id in job_array:
        rim_radius = parameters["rim_radius"]
        mass = parameters["mass"]

        # metrics for the drawbar sim
        avg_dcs = []
        z_variances = []
        sinkages = []
        local_job_folder = "/data/wheel_sim_pipeline_data/automation_test_" + drawbar_job_id
        os.makedirs(local_job_folder, exist_ok = True)
        for slip in slip_values_str:
            os.makedirs(local_job_folder+"/SkidSteerSim_"+slip,exist_ok=True)
        for slip in slip_values_str:
            subprocess.run(["./automated_pipeline.sh", "download", drawbar_job_id, slip])
            df = pd.read_csv('/data/wheel_sim_pipeline_data/automation_test_'
                             + drawbar_job_id + '/SkidSteerSim_' + slip +
                             '/output.csv')
            avg_dcs.append((df["f_x"] / df["f_z"]).mean())
            z_variances.append(df["pos_z"].var(ddof=0)) # leaving blank does unbiased variance
            sinkages.append(df["pos_z"].min())
        # area under curve of drawbar pulls
        slip1 = None
        slip2 = None
        for i in range (len(avg_dcs) - 1):
            p1 = avg_dcs[i]
            p2 = avg_dcs[i + 1]
            if(p1 == 0):
                x_inter = slip_values[i]
                break
            if(p2 == 0):
                x_inter = slip_values[i+1]
                break
            if(p1 < 0 < p2):
                slip1 = slip_values[i]
                slip2 = slip_values[i + 1]
                point1 = p1
                point2 = p2
                slope = (point2 - point1)/(slip2 - slip1)
                b = point1 - (slope * slip1)
                x_inter = -b/slope
                break
        if(slip1 is None):
            print(f"no 0 intercept found")
            x_inter = 0

        #metrics for the steering test
        avg_f_y = []
        local_job_folder = "/data/wheel_sim_pipeline_data/automation_test_" + steering_job_id
        os.makedirs(local_job_folder, exist_ok = True)
        for slip in slip_values_str:
            os.makedirs(local_job_folder+"/Steering_"+slip,exist_ok=True)
        for slip in slip_values_str:
            subprocess.run(["./automated_pipeline.sh", "download", steering_job_id, slip])
            df = pd.read_csv('/data/wheel_sim_pipeline_data/automation_test_'
                             + steering_job_id + '/Steering_' + slip +
                             '/output.csv')
            avg_f_y.append((df["f_y"]).mean())

        # perform horizontal shift
        slips_shifted = [s - x_inter for s in slip_values]
        #area under curve of zero crossing
        dcs_under_curve = np.trapezoid(avg_dcs, slips_shifted)
        if np.isnan(dcs_under_curve):
            dcs_under_curve = 0.1
        else:
            dcs_under_curve *= 10
        # MASS
        # mass = parameters["mass"]
        z_var = float(np.mean(z_variances))
        sinkage = float(np.mean(sinkages)) - rim_radius + 0.41
        mean_f_y = np.mean(avg_f_y)
        # compute score with weighted averages
        # score = dcs_under_curve * 0.75 + mass * 0.1 + z_var * 0.05 + 0.1 * sinkage
        score = dcs_under_curve * 0.55 + mass * 0.2 +sinkage * 0.1 + mean_f_y * .1 + z_var * 0.05 

        if np.isnan(score):
            for _ in range(3):
                print("SCORE WAS NAN")
            score = 0.1
        # Set raw_data as a dictionary with metric names as keys and results as values
        raw_data = {metric_name: score}

        print("dcs: ",dcs_under_curve, " z_var: ",z_var," sinkage: ",sinkage)

        # Complete the trial with the result
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        print(f"Completed trial {trial_index} with {raw_data=}")

    summary = client.summarize()
    summary["job_id"] = all_job_ids
    summary.to_csv("data.csv", index=False)

best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

# saving the dataframe
summary = client.summarize()
summary["job_id"] = all_job_ids
summary.to_csv("data.csv", index=False)

x = summary["rim_radius"]
y = summary["score"]
plt.scatter(x, y, c="blue", alpha=0.3, marker="o", zorder=3000)
plt.xlabel("rim_radius")
plt.ylabel("Score")
plt.title("Simulation Evaluation")
plt.legend()
plt.show()
