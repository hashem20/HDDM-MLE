import hddm
import pandas as pd
import numpy as np

data1 = hddm.load_csv('data.csv')
n = 87
##### First Model
model1 = hddm.HDDM(data1, bias = False)
model1.find_starting_values()
model1.sample(5000, burn=1000)
output1 = model1.gen_stats()

variables = ['a_subj', 'v_subj', 't_subj']
columns = ['a', 'v', 't']
params1 = pd.DataFrame(np.empty([n, len(columns)]), columns=columns)
counter = np.zeros(len(columns), dtype=int)
for i, a in enumerate(output1.index):
    for j, b in enumerate(variables):
        if b in a:
            params1.iloc[counter[j], j] = output1.iloc[i,0]
            counter[j] += 1
#### random walk - generating new rt's and responses based on first parameters
data2 = data1.copy()
dt = .01
c = 1

for i in range(len(data2)):
    idx = int(data2['subj_idx'].iloc[i]) - 1
    v = params1['v'][idx] 
    a = params1['a'][idx]
    t = params1['t'][idx] 
    
    y = .5 * a
    flag = True
    k = 1
    
    while flag:
        dW = np.sqrt(dt) * np.random.randn()
        dy = v * dt + c * dW
        y += dy
        if (y > a) | (y < 0):
            flag = False
        k += 1
        
    rt = t + (k-1) * dt
    response = (1+np.sign(y))/2

    data2['response'][i] = response
    data2['rt'][i] = rt
    
    del y
###### Second Model
model2 = hddm.HDDM(data2, bias = False)
model2.find_starting_values()
model2.sample(5000, burn=1000)
output2 = model2.gen_stats()

params2 = pd.DataFrame(np.empty([n, len(columns)]), columns=columns)
counter = np.zeros(len(columns), dtype=int)
for i, a in enumerate(output2.index):
    for j, b in enumerate(variables):
        if b in a:
            params2.iloc[counter[j], j] = output2.iloc[i,0]
            counter[j] += 1

corr = np.zeros(len(columns))
for i in range(len(corr)):
    r = np.corrcoef(params1.iloc[:,i], params2.iloc[:,i])
    corr[i] = r[0][1]
corrs = pd.DataFrame([params1.columns.values, corr]).T

