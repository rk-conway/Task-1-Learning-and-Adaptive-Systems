import pandas as pd
import numpy as np

def entropy(col):
    _, counts = np.unique(col, return_counts=True)
    total = sum(counts)
    ent = 0

    for c in counts:
        p = c / total
        ent -= p * np.log2(p)

    return ent

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])

    weighted_entropy = 0

    for v in data[feature].unique():
        subset = data[data[feature] == v]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

# Dataset 
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    
    'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                 'High','Normal','Normal','Normal','High','Normal','High'],
    
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
             'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes',
                   'No','Yes','Yes','Yes','Yes','Yes','No']
})

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

for f in features:
    print(f, ":", round(information_gain(data, f, 'PlayTennis'), 3))
