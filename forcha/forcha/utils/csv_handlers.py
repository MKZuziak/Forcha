import os
import csv

def save_coalitions(values: dict,
                    path,
                    name: str,
                    iteration: int,
                    mode: int = 0):
    if mode == 0:
        with open(os.path.join(path, name), 'a+', newline='') as csv_file:
            field_names = ['coalition', 'value', 'iteration']
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(field_names)
            for col, value in values.items():
                csv_writer.writerow([col, value, iteration])
    else:
        with open(os.path.join(path, name), 'a+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for col, value in values.items():
                csv_writer.writerow([col, value, iteration])