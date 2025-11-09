#!/usr/bin/env python3
"""Run MTEB evaluation on a local model."""
import json
from sentence_transformers import SentenceTransformer, models
import mteb
from mteb.cache import ResultCache
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = 'marksverdhei/mimir-embed-500m-1epoch'
model_path = 'marksverdhei/mimir-embed-3000'
print(AutoModelForCausalLM.from_pretrained(model_path))

tokenizer = AutoTokenizer.from_pretrained("marksverdhei/mimir-embed-3000")
model = models.Transformer(model_path)

pooling = models.Pooling(1024, pooling_mode="lasttoken")
print(pooling)

model = SentenceTransformer(modules=[model, pooling])

model.model_card_data.model_name = "local/eot-token2"

# Get tasks filtered by language
tasks = mteb.get_tasks(
    tasks=["NorQuadRetrieval"],
    languages=["nob"]
)

# Run evaluation
results = mteb.evaluate(
    model,
    tasks=tasks,
    cache=ResultCache("results")
)

print("Evaluation complete!")
print(f"Results saved to: results/")

print(results)
