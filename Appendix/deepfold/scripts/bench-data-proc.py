import re
import os
import csv

directory = './outputs'
pattern = r'(.*)time:\s*\[\d+\.\d+ (µs|ms|s) (\d+\.\d+ (µs|ms|s)) \d+\.\d+ (µs|ms|s)\]'
table = dict()

for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    filename = file.split(".")[0]
    if os.path.isdir(file_path):
        continue
    with open(file_path, 'r') as f:
        text = f.read()
    matches = re.findall(pattern, text)
    for m in matches:
        ind = m[0].strip()
        time = m[2]
        table[ind] = time
    write_file = './outputs/' + filename + '-benchemarks.csv'
    with open(write_file, mode='w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(['item', 'time'])
        for k in table:
            writer.writerow([k, table[k]])