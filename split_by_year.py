# import csv
# import os

# input_file = "Seoul_dataset.csv"
# file_handles = {}
# csv_writers = {}

# try:
#     with open(input_file, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         header = next(reader)
        
#         for row in reader:
#             if not row: continue
#             date_str = row[0]
#             year = date_str[:4]
            
#             if year not in file_handles:
#                 output_file = f"Seoul_dataset_{year}.csv"
#                 file_handles[year] = open(output_file, 'w', encoding='utf-8', newline='')
#                 csv_writers[year] = csv.writer(file_handles[year])
#                 csv_writers[year].writerow(header)
            
#             csv_writers[year].writerow(row)
            
#     print("Successfully split the dataset by year:")
#     for year in sorted(file_handles.keys()):
#         print(f"- Seoul_dataset_{year}.csv")
        
# finally:
#     for f in file_handles.values():
#         f.close()
