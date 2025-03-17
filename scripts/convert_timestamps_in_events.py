#

import json
from datetime import datetime


with open('events_temp.json', 'r') as f:
    d = json.load(f)

for item in d:
    t1 = datetime.strptime(item['t1'], "%Y-%m-%d %H:%M:%S.%f")
    t2 = datetime.strptime(item['t2'], "%Y-%m-%d %H:%M:%S.%f")
    
    item.update({
        't1': t1.second + t1.microsecond / 1E6, 
        't2': t2.second + t2.microsecond / 1E6})

with open('events_temp2.json', 'w') as f:
    json.dump(d, f, indent=4)