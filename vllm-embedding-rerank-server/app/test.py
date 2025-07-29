import heapq

scores=[0.1,0.3,0.5]
documents=['a','b','c']
all_pairs = [
    {'id': index+1, 'score': score, 'document': documents[index]} 
    for index, score in enumerate(scores)
]
all_pairs = heapq.nlargest(2, all_pairs, key=lambda x: x['score'])
print(all_pairs)