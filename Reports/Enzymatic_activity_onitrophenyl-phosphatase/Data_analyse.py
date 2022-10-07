import os
import csv

filename = 'data.txt'

dirname = os.path.dirname(__file__)
file = os.path.join(dirname, filename)

with open(file, 'r') as f:
    data = csv.reader(f, delimiter=' ')
    for row in data:
        print(row)
    f.close()

