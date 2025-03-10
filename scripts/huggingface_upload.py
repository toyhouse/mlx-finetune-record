from huggingface_hub import HfApi, upload_folder
import os
import sys

def upload_model(local_path, repo_id):
    try:
        # Ensure the repository exists
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        
        # Upload the folder
        upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            create_pr=False
        )
        print(f"Successfully uploaded model to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python huggingface_upload.py /path/to/model Lckoo1230/repo-name")
        sys.exit(1)
    
    local_path = sys.argv[1]
    repo_id = sys.argv[2]
    
    success = upload_model(local_path, repo_id)
    sys.exit(0 if success else 1)
