import pandas as pd
from apyori import apriori

dataset = pd.read_csv('apriori_data.csv', header = None, error_bad_lines=False)

records = []
for i in range(10):
    records.append([str(dataset.values[i, j]) for j in range(4)])

association_rules = apriori(records, min_support=0.2, min_confidence=0.2, min_lift=2, min_length=2)  

for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    print('Rule: ' +items[0]+ ' -> ' +items[1])
    print('Support: ' +str(item[1]))
    print('Confidence: ' +str(item[2][0][2]))
    print('Lift: ' +str(item[2][0][3]))