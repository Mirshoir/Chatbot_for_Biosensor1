import os

# Define the directory structure and files
structure = {
    "avicenna_chatbot": {
        "backend": [
            "app.py",
            "chatbot_logic.py",
            "vision_handler.py",
            "biometric_reader.py",
            "chroma_store.py",
            "config.py"
        ],
        "frontend": [
            "index.html",
            "style.css",
            "script.js"
        ]
    }
}

def create_structure(base_path, tree):
    for folder, contents in tree.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for subfolder_or_file in contents:
            if isinstance(subfolder_or_file, dict):
                create_structure(folder_path, subfolder_or_file)
            else:
                file_path = os.path.join(folder_path, subfolder_or_file)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("")  # create empty file

if __name__ == "__main__":
    base_dir = os.getcwd()  # or use custom path like "/your/target/path"
    create_structure(base_dir, structure)
    print("Project structure created successfully.")
