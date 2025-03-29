import json
import csv
import os

# Specify the directory containing your JSON files
directory = './target/criterion'
csv_filename = 'criterion_results.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['name', 'Slope_Point_Estimate'])

    # Loop through each file in the directory
    for foldername in os.listdir(directory):
        folder = os.path.join(directory, foldername, "new")
        file_path = os.path.join(folder, "estimates.json")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        slope_point_estimate = data['mean']['point_estimate']

        file_path = os.path.join(folder, "benchmark.json")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        name = data['group_id']
        writer.writerow([name, slope_point_estimate])
