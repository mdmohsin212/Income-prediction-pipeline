import os
os.makedirs("src", exist_ok=True)

files = [
    "data_ingestion.py",
    "data_preprocessing.py",
    "feature_engineering.py",
    "model_building.py",
    "model_evaluation.py"
]

for file in files:
    path = os.path.join("src", file)
    if not os.path.exists(path):
        open(path, "w").close()

print("Project structure created.")