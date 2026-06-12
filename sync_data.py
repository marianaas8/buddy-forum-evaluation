"""
Data Synchronisation — Buddy Forum Server
==========================================
Downloads habit metadata and GLB model files from the Buddy Forum server,
and fetches user interaction metrics (likes, downloads, message ratings).

Outputs written to data/:
  eval_list.csv         — habit name, prompt, creator, date, GLB path, animation command
  metrics.json          — habit interaction events (model_rating, model_download)
  message_metrics.json  — Buddy response rating events

Configuration:
  SERVER_URL — set to the Buddy Forum server IP and port
"""
import requests
import os
import pandas as pd
import json

SERVER_URL = "http://YOUR_SERVER_IP:8080"  # Replace with your Buddy Forum server IP and port
LOCAL_MODELS_DIR = "data"  # Directory to store downloaded models
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

def auto_sync():
    print("Searching for new models ...")
    response = requests.get(f"{SERVER_URL}/api/habits_sync")
    response.raise_for_status()  # Check if the request was successful
    habits_data = response.json()

    eval_list = []
    for h in habits_data:
        filename = h.get('object')

        # strip animations from filename if present (e.g. "model.glb ||| walk")
        anim_command = "none"
        if "|||" in filename:
            parts = filename.split("|||")
            filename = parts[0].strip()
            if len(parts) > 1:
                anim_command = parts[1].strip()
                
        if not filename:
            print(f"Skipping entry with missing 'object': {h}")
            continue

        local_glb_path = os.path.join(LOCAL_MODELS_DIR, filename)

        if not os.path.exists(local_glb_path):
            print(f"Downloading {filename} ...")
            glb_url = f"{SERVER_URL}/generated_models/{filename}"
            r = requests.get(glb_url)
            if r.status_code == 200:
                with open(local_glb_path, "wb") as f:
                    f.write(r.content)

        # Extract prompt from description
        full_desc = h.get('description', 'N/A')
        parts = full_desc.split(' ||| Prompt: ')
        prompt = parts[1].strip() if len(parts) > 1 else parts[0].strip()

        eval_list.append({
            'Nome': h.get('name', 'N/A'),
            'Prompt': prompt,
            'Criador': h.get('creator', 'N/A'),
            'Data': h.get('createdAt', h.get('created_at', 'N/A')),
            'Objeto': local_glb_path,
            'Animacao': anim_command
        })

    df = pd.DataFrame(eval_list)
    df.to_csv("data/eval_list.csv", index=False, encoding='utf-8')
    print("Data synchronization complete. Eval list saved to data/eval_list.csv")

    print("Syncing user metrics (likes, downloads) ...")
    metrics_response = requests.get(f"{SERVER_URL}/api/metrics")
    if metrics_response.status_code == 200:
        text = metrics_response.text
        try:
            data = metrics_response.json()
        except ValueError:
            print("⚠️  Metrics endpoint returned invalid JSON")
            data = None
        # basic schema check
        if isinstance(data, list) and data:
            sample = data[0]
            if not ('modelName' in sample and 'eventType' in sample):
                print("⚠️  Warning: metrics payload doesn't look like habit event data.")
                print(f"         sample keys = {list(sample.keys())[:10]}")
                print("         Ensure the server /api/metrics endpoint returns the expected events format.")
        with open(os.path.join(LOCAL_MODELS_DIR, "metrics.json"), "w", encoding='utf-8') as f:
            f.write(text)
        print("User metrics synced successfully to data/metrics.json")
    else:
        print(f"Failed to sync user metrics. Status code: {metrics_response.status_code}")

    # Also fetch message metrics separately
    print("Syncing message metrics (buddy response ratings) ...")
    msg_metrics_response = requests.get(f"{SERVER_URL}/api/metrics/messages")
    if msg_metrics_response.status_code == 200:
        msg_text = msg_metrics_response.text
        try:
            msg_data = msg_metrics_response.json()
        except ValueError:
            print("⚠️  Message metrics endpoint returned invalid JSON")
            msg_data = None
        with open(os.path.join(LOCAL_MODELS_DIR, "message_metrics.json"), "w", encoding='utf-8') as f:
            f.write(msg_text)
        print("Message metrics synced successfully to data/message_metrics.json")
    else:
        print(f"Note: Message metrics endpoint not available. Status code: {msg_metrics_response.status_code}")


if __name__ == "__main__":
    auto_sync()