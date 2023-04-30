import torch

import numpy as np

from sentence_transformers import SentenceTransformer


device = torch.device('cuda')


model = SentenceTransformer('all-mpnet-base-v2')

#Our sentences we like to encode
sentences = np.array([
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.'
    ])

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences, convert_to_tensor=True, device='cuda')

# cosines
cosines = []

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

    cosines.append(torch.cosine_similarity(embeddings[0], embedding, dim=0))

print(type(embeddings))
print(model)
print(type(model))
print(embeddings.shape)

print(cosines)




