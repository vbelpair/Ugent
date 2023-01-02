import openpyxl
import numpy as np
import matplotlib.pyplot as plt

pathdir = "C:\\Users\\Tech\\OneDrive\\Ugent vincent\\2022-2023\\QC&TA\\report"

path_sc = pathdir + "\\NTA - rEV scatter 29.11.2022.xlsx"
path_fl = pathdir + "\\NTA - rEV fluorescent 29.11.2022.xlsx"

path = path_sc

wb_obj = openpyxl.load_workbook(path)

sheet_obj = wb_obj.active

r1 = 51
r2 = sheet_obj.max_row

cells = sheet_obj[f'A{r1}':f'F{r2}']

x = np.zeros(r2-r1+1)
y1 = np.zeros(r2-r1+1)
y2 = np.zeros(r2-r1+1)
y3 = np.zeros(r2-r1+1)
mean = np.zeros(r2-r1+1)
std = np.zeros(r2-r1+1)

i = 0
for c1, c2, c3, c4, c5, c6 in cells:

    x[i] = c1.value
    y1[i] = c2.value
    y2[i] = c3.value
    y3[i] = c4.value
    mean[i] = c5.value
    std[i] = c6.value
    i += 1

plt.figure(figsize=(15,10))
plt.plot(x,y1, label='video 1')
plt.plot(x,y2, label='video 2')
plt.plot(x,y3, label='video 3')
plt.plot(x,mean, label='mean', ls='--', color='black')
plt.fill_between(x,mean-std,mean+std, alpha=0.4, label='std')
plt.xlabel('Bin centre (nm)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Concentration (particles/ml', fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.savefig(pathdir + "\\fig_scatter.png")