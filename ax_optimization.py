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
        name="rim_radius", parameter_type="float", bounds=(0, 1)
    ),
    RangeParameterConfig(
        name="width", parameter_type="float", bounds=(0, 1)
    ),    
    RangeParameterConfig(
        name="grouser_number", parameter_type="float", bounds=(0, 1)
    ),
    RangeParameterConfig(
        name="grouser_thickness", parameter_type="float", bounds=(0, 1)
    ),
        RangeParameterConfig(
        name="grouser_height", parameter_type="float", bounds=(0, 1)
    ),
    RangeParameterConfig(
        name="control_point_deviation", parameter_type="float", bounds=(0, 1)
    ),
        RangeParameterConfig(
        name="wave_number", parameter_type="float", bounds=(0, 1)
    ),
    RangeParameterConfig(
        name="wave_amplitude", parameter_type="float", bounds=(0, 1)
    )
]

client.configure_experiment(parameters=parameters)
test_arrays = ["32720007", "32781190", "33283836", "33283810", "32461246", "32724206", "32718125", "32718537", "32721460"]
metric_name = "score" # this name is used during the optimization loop in Step 5
objective = f"{metric_name}" # minimization is specified by the negative sign
rim_radius = parameters["rim_radius"]
client.configure_optimization(objective=objective)


for batch_number in range(3): # Run 10 rounds of trials
    # We will request three trials at a time in this example
    trials = client.get_next_trials(max_trials=3)

    #get batch parameters
    job_array = []
    for trial_index, parameters in trials.items():
        job_array.append([trial_index, parameters, 0])

    #launch every job
    for i in range(len(job_array)):
        trial_index, parameters, _ = job_array[i]
        wheel_param = {
            "rim_radius": parameters["rim_radius"],
            "width": parameters["width"],
            "grouser_number": parameters["grouser_number"],
            "grouser_thickness": parameters["grouser_thickness"],
            "grouser_height": parameters["grouser_height"],
            "control_point deviation": parameters["control_point_deviation"],
            "wave_number": parameters["wave_number"],
            "wave_amplitude": parameters["wave_amplitude"],
            "outer_radius": parameters["rim_radius"] + parameters["grouser_height"]
        }
        # download wheel json
        with open("wheel_jsons/wheel_parameters.json", "w") as json_file:
            json.dump(wheel_param, json_file, indent=4)

        # create json w/ job parameters
        job_json = {
            "terrain_filepath": "/jet/home/matthies/moonranger_mobility/terrain/GRC_3e5_Reduced_Footprint.csv",
            "wheel_folder_path": "/jet/home/matthies/moonranger_mobility/meshes/wheel_"+str(trial_index)+"/",
            "data_drivepath": "/ocean/projects/mch240013p/matthies/",
            "slip": 0.2, #this should be in the automation.sh on onDemand
            "sim_endtime": 0.2,
            "output_dir": ""
        }
        # download job json
        with open("job_json/job_parameters.json", "w") as json_file:
            json.dump(job_json, json_file, indent=4)

        # bash script creates wheel mesh, uploads mesh & json, and launches script
        # print("start bash script")
        # script_result = subprocess.run(["./automated_pipeline.sh", "upload", str(trial_index)], capture_output=True, text=True)
        # JOB_ID = 0
        # if script_result.returncode == 0:
        #     output_lines = script_result.stdout.strip().split('\n')
        #     for line in output_lines:
        #         if "job id:" in line:
        #             JOB_ID = line.split(":")[1]
        #     if JOB_ID == 0:
        #         print("job id not found")
        #     else:
        #         print("Script success and jobid captured "+ JOB_ID)
        # else:
        #     print("Script failed or filename not capture")
        # print("end bash script")
        # print("stdout:")
        # print(script_result.stdout)
        # print("stderr:")
        # print(script_result.stderr)

        job_array[i][2] = test_arrays[trial_index]


    print("start gmail monitor")
    job_ids = [row[2] for row in job_array]
    gmail_monitor.monitor(job_ids)
    # slip values currently tested
    slip_values = [-0.1, 0.0, 0.1, 0.3]
    for trial_index, parameters, job_id in job_array:

        # average drawbar pulls for each slip
        avg_dcs = []
        z_variances = []
        sinkages = []
        for slip in slip_values:
            df = pd.read_csv('~/Documents/wheel_sim_data/automation_test_' 
                             + job_id + '/SkidSteerSim_' + slip + 
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
                slip1 = slip2 = slip_values[i]
                point1 = point2 = 0.0
                break
            if(p2 == 0):
                slip1 = slip2 = slip_values[i + 1]
                point1 = point2 = 0.0
                break
            if(p1 < 0 < p2):
                slip1 = slip_values[i]
                slip2 = slip_values[i + 1]
                point1 = p1
                point2 = p2
                break

        if(slip1 is None):
            print(f"no 0 intercept found")
        else:
            slope = (point2 - point1)/(slip2 - slip1);
            b = point1 - (slope * slip1)
            x_inter = -b/slope
        # perform horizontal shift
        slips_shifted = [s - x_inter for s in slip_values]
        #area under curve of zero crossing
        dcs_under_curve = np.trapz(avg_dcs, slips_shifted)

        # MASS
        mesh_file = "~/Documents/wheel_sim_pipeline/wheel.obj"
        # run Blender headless to compute mass
        proc = subprocess.run([
            "blender",
            "--background",
            "--python", "compute_volume.py",
            "--", mesh_file, "2.7"           # density = 2.7 g/cm³
        ], capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"Volume calc failed:\n{proc.stderr}")

        mass = float(proc.stdout.strip())
        z_var = float(np.mean(z_variances))
        sinkage = float(np.mean(sinkages)) - rim_radius - 0.41
        # compute score with weighted averages
        score = dcs_under_curve * 0.75 + mass * 0.1 + z_var * 0.05 + 0.1 * sinkage
        # Set raw_data as a dictionary with metric names as keys and results as values
        raw_data = {metric_name: score}

        # Complete the trial with the result
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        print(f"Completed trial {trial_index} with {raw_data=}")

        

best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

# display=True instructs Ax to sort then render the resulting analyses
# cards = client.compute_analyses(display=False)
# print(cards)
# for card in cards:
#      print(card)

# saving the dataframe
summary = client.summarize()
summary.to_csv("data.csv", index=False)
print(type(summary))

x = summary["rim_radius"]
y = summary["score"]

plt.scatter(x, y, c="blue", alpha=0.3, marker="o", zorder=3000)

plt.xlabel("rim_radius")
plt.ylabel("Score")
plt.title("Simulation Evaluation")
plt.legend()
plt.show()