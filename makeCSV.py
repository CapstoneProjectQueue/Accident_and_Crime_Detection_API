import csv
import os

filename = "data.csv"
file_path1 = "C:\\Users\\USER\\Desktop\\capstonequeue\\img\\abnormal"
file_path2 = "C:\\Users\\USER\\Desktop\\capstonequeue\\img\\normal"
total_files1 = 1080
total_files2 = 2166

# Create the CSV file
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(['ClassId', 'File Name', 'Path'])

    for i in range(1, total_files1 + 1):
        file_name = f"{i}.jpg"
        full_path = os.path.join(file_path1, file_name)
        writer.writerow(['0', file_name, full_path])

    for i in range(total_files1 + 1, total_files2 + 1):
        file_name = f"{i}.jpg"
        full_path = os.path.join(file_path2, file_name)
        writer.writerow(['1', file_name, full_path])
