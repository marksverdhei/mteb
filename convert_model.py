from sentence_transformers import SentenceTransformer, models
import json

# Load transformer
transformer = models.Transformer('marksverdhei/mimir-embed-500m-1epoch')

# Load pooling config from JSON file
with open('/media/me/storage/Models/hf_repos/mimir-embed-500m-1epoch/1_Pooling/config.json', 'r') as f:
    pooling_config = json.load(f)

# Create pooling module
pooling = models.Pooling(**pooling_config)

# Combine into SentenceTransformer
model = SentenceTransformer(modules=[transformer, pooling])

model.encode("Hei på deg")

import IPython; IPython.embed()
# model.("Hei på deg")
# model.save_pretrained("/media/me/storage/Models/hf_repos/mimir-embed-500m-1epoch/")

# SentenceTransformer("NbAiLab/mimir-mistral-500m-core-scratch").save_pretrained("mimir-baseline")
