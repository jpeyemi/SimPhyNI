import os
import sys
import numpy as np
from numpy import matlib
from scipy import stats
import pandas as pd

filename = sys.argv[-1]
file = pd.read_csv(filename, index_col = False)
# file.round()
newdir = '1-Inputs'
if not os.path.exists(newdir):
	os.mkdir(newdir)
os.chdir(newdir)
cols  = file.columns.values.tolist()
ind = cols[0]
cols = cols[1:]
lst = []
for c in cols:
	dur = f'{c}'
	if not os.path.exists(dur):
		os.mkdir(dur)
	df = file[[ind,c]].copy()
	df.loc[df[c] < .5, c] = np.nan
	df.loc[df[c] >= .5, c] = 1
	# df[c] = df[c].div(df[c])
	df[c] = df[c].round(0)
	df.fillna(0, inplace = True)
	df.rename(columns = {ind:'sample',c:'syst'}, inplace = True)	
	df = df.astype({'syst':int})
	df.to_csv(f'{dur}/syst.csv',index= False)
	lst.append(c)

file = open('systems.txt','w')
for item in lst:
	file.write(item+"\n")
file.close()


