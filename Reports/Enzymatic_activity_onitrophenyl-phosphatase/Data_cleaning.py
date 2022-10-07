import os

#================
# Function setup
#================

def convert_txt_to_grid(txtdata, minlen=0):
    lines = txtdata.split('\n')
    grid = []
    for line in lines:
        l = line.split()
        if len(l) >= minlen:
            grid.append(list(filter(lambda x: x != '', l)))
    return grid

def find_tables(raw_data, dim, xlabel0, ylabel0):
    N, M = dim[0], dim[1]
    grid = convert_txt_to_grid(raw_data, M)
    H = len(grid) # hight of grid
    searching, i = True, 0
    tables = []
    while searching:
        if (grid[i][0] == xlabel0) & (grid[i+1][0] == ylabel0):
            table = grid[i+1:N+i+1]
            tables.append(table)
            i += N
        i += 1
        searching = H-i > N
    return tables

def extract_tables_from_txt(filename, tablenames=None):
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, filename)
    with open(file, 'r') as f:
        raw_data = f.read()
        f.close()
    tables = find_tables(raw_data, (8, 12), '1', 'A')
    if tablenames:
        table_dict = {}
        for i, table in enumerate(tables):
            table_dict[tablenames[i]] = table
        tables = table_dict
    return tables

def write_tables(tables, filename, columnlabels=None):
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, filename)
    with open(file, 'w') as f:
        if columnlabels:
            f.write(columnlabels)
        for name, table in tables.items():
            for row in table:
                f.write(' '.join(row) + ' ' + name + '\n')
        f.close()

#===========
# Execution
#===========

filenames = ['DataStandardCurve.txt', 'Vincent_Lucas_2.txt', 'Céline_Leen_3.txt']
tablenames = [['C0', 'M1'], ['M2'], ['M3']]
table_dict = {}
for i, filename in enumerate(filenames):
    table_dict.update(extract_tables_from_txt(filename, tablenames[i]))

write_tables(table_dict, 'data.txt', columnlabels='rowlabel ' + ' '.join( [str(i) for i in range(1, 13)] ) + ' label\n')