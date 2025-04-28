import pandas as pd
import numpy as np

# ✅ Sample Dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport']
df = pd.DataFrame(data, columns=columns)

# -------------------------------
# ✅ Find-S Algorithm
# -------------------------------
def find_s(df):
    concepts = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    
    hypothesis = ['0'] * len(concepts[0])

    for i, val in enumerate(concepts):
        if target[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = val[j]
                elif hypothesis[j] != val[j]:
                    hypothesis[j] = '?'
    return hypothesis

# -------------------------------
# ✅ Candidate Elimination Algorithm
# -------------------------------
def candidate_elimination(df):
    concepts = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values

    def consistent(hypothesis, instance):
        for x, y in zip(hypothesis, instance):
            if x != '?' and x != y:
                return False
        return True

    n_features = len(concepts[0])
    S = ['0'] * n_features
    G = [['?' for _ in range(n_features)]]

    for i, instance in enumerate(concepts):
        if target[i] == 'Yes':
            for j in range(n_features):
                if S[j] == '0':
                    S[j] = instance[j]
                elif S[j] != instance[j]:
                    S[j] = '?'
            G = [g for g in G if consistent(g, instance)]
        else:
            G_temp = []
            for g in G:
                for j in range(n_features):
                    if g[j] == '?':
                        if S[j] != '?':
                            new_hyp = g.copy()
                            new_hyp[j] = S[j]
                            if not consistent(new_hyp, instance):
                                G_temp.append(new_hyp)
            G = G_temp

    return S, G

# -------------------------------
# ✅ Run Algorithms and Print Results
# -------------------------------
print("🎯 Training Data:\n")
print(df)

print("\n🔍 Running Find-S Algorithm...")
find_s_hypothesis = find_s(df)
print("Final Hypothesis (Find-S):", find_s_hypothesis)

print("\n🔍 Running Candidate Elimination Algorithm...")
S_final, G_final = candidate_elimination(df)
print("Final Specific Hypothesis (S):", S_final)
print("Final General Hypotheses (G):")
for g in G_final:
    print(g)
