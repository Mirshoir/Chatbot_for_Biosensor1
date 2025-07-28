import json
import os

TEMPLATE_FILE = "prompt_templates.json"

def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        return {"cognitive_load": "", "user_specific": ""}
    with open(TEMPLATE_FILE, "r") as f:
        return json.load(f)

def save_templates(templates):
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(templates, f, indent=2)
