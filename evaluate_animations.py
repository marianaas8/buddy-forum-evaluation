"""
Phase 2 Evaluation — Animation Quality
========================================
Evaluates skeletal animations assigned to habits via Tripo AI's rig+retarget pipeline.

Inputs  (via sync_data.py):
  data/eval_list.csv      — habit name, GLB path, animation command
  data/metrics.json       — user interaction events (likes, downloads)

Outputs:
  data/anim_results.csv   — per-model animation quality metrics
  MLflow run              — aggregated metrics under experiment
                            'habit_animation_evaluation'

Metrics (skeletal animations only):
  Geometry  — max edge expansion %, mesh clipping ratio, volume variation %,
               edge stretch variance
  Motion    — foot skating ratio, jitter score (L2 acceleration variance)
  Structure — watertight survival (base pose → animated)
  Semantic  — CLIP score (mid-frame render vs. animation command), ImageReward
  User      — likes, dislikes, downloads

Habits with UI-only animations (float, bounce, spin, pulse) are recorded but skip
Blender evaluation.

Configuration:
  BLENDER_EXEC        — update to local Blender executable path
  MLFLOW_TRACKING_URI — update to your MLflow server
  USE_MLFLOW          — set False to skip MLflow logging
"""
import os
import csv
import json
import subprocess
import contextlib
from datetime import datetime
from statistics import mean

import pandas as pd
import torch
import clip
from PIL import Image

try:
    import ImageReward as RM
    HAS_IMAGE_REWARD = True
except ImportError:
    HAS_IMAGE_REWARD = False

USE_MLFLOW = True
if USE_MLFLOW: import mlflow

MLFLOW_TRACKING_URI = "http://YOUR_MLFLOW_SERVER:5000"
EXPERIMENT_NAME = "habit_animation_evaluation"
BLENDER_EXEC = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" 
RUN_PARAMS = {
    "model_provider": "Tripo AI",
    "model_version": "",
    "task_type": "text_to_model, animate_prerigcheck, animate_rig, animate_retarget",    
}

@contextlib.contextmanager
def optional_mlflow_run(run_name):
    
    if USE_MLFLOW:
        with mlflow.start_run(run_name=run_name) as run: yield run
    else:
        print(f"\n🚀 [TEST MODE] Starting Run: {run_name}")
        yield None

def log_metric_safe(key, value):
    if USE_MLFLOW: mlflow.log_metric(key, float(value))

# SkeletalAnimationEvaluator class encapsulates the logic for evaluating skeletal animations using Blender and AI models (CLIP and ImageReward).
class SkeletalAnimationEvaluator:
    def __init__(self):
        print("Initializing AI Models (CLIP & ImageReward)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        if HAS_IMAGE_REWARD:
            self.reward_model = RM.load("ImageReward-v1.0", device=self.device) # ImageReward-v1.0 is a model that scores how well an image matches a text prompt, trained on human preferences. 
        print("Models Loaded.\n" + "="*40)

    # Evaluate a GLB file using Blender to extract animation quality metrics. It runs a Blender Python script and captures the output metrics.
    def evaluate_glb_with_blender(self, glb_path):
        abs_glb_path = os.path.abspath(glb_path)
        output_json = abs_glb_path.replace(".glb", "_anim_metrics.json")
        metrics = {"max_edge_expansion_pct": 0.0, "foot_skating_ratio": 0.0, "jitter_acceleration": 0.0, "is_watertight_base": False, "is_watertight_anim": False, "render_path": "", "valid_rig": False}
        
        cmd = [BLENDER_EXEC, "--background", "--python", "blender_anim_eval.py", "--", abs_glb_path, output_json]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    result = json.load(f)
                    if "error" not in result:
                        metrics.update(result)
                        metrics["valid_rig"] = True
                    else:
                        metrics["error_msg"] = result["error"]
                os.remove(output_json)
        except subprocess.CalledProcessError: pass
        return metrics

    # Evaluate the animation quality metrics using AI models (CLIP and ImageReward) with the middle frame of the animation.
    def run_ai_metrics(self, render_path, anim_command):
        ai_metrics = {"clip_action_score": 0.0, "image_reward_score": 0.0}
        if os.path.exists(render_path):
            try:
                pil_img = Image.open(render_path).convert("RGB")
                img_prep = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                
                action_prompt = f"a 3D model performing the action: {anim_command}"
                text_tokens = clip.tokenize([action_prompt]).to(self.device)
                
                with torch.no_grad():
                    img_features = self.clip_model.encode_image(img_prep)
                    text_features = self.clip_model.encode_text(text_tokens)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    clip_score = (img_features @ text_features.T).item()
                ai_metrics['clip_action_score'] = round(clip_score, 4)
                
                if HAS_IMAGE_REWARD:
                    ai_metrics['image_reward_score'] = round(self.reward_model.score(action_prompt, pil_img), 4)
            except Exception as e: print(f"  [Error] AI Eval failed: {e}")
        return ai_metrics

def load_user_metrics():
    metrics_path = "data/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except: return []
    return []

def load_message_metrics():
    message_metrics_path = "data/message_metrics.json"
    if os.path.exists(message_metrics_path):
        with open(message_metrics_path, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except json.JSONDecodeError as e: print(f"⚠️  Failed to parse message_metrics.json: {e}")
    return []

def run_animation_evaluation(input_csv="data/eval_list.csv", output_csv="data/anim_results.csv"):
    if not os.path.exists(input_csv): return
    records = []
    with open(input_csv, mode='r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
        
    user_metrics = load_user_metrics()
    message_metrics = load_message_metrics()
    evaluator = SkeletalAnimationEvaluator()
    results = []
    
    ui_animations = ['float', 'bounce', 'spin', 'pulse']
    ui_count = 0
    skeletal_count = 0

    if USE_MLFLOW: mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); mlflow.set_experiment(EXPERIMENT_NAME)
    
    with optional_mlflow_run("Anim_Eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")):
        for index, record in enumerate(records):
            name = record.get('Nome', f'Habit_{index}')
            glb_path = record.get('Objeto', '')
            anim_command = record.get('Animacao', 'none').strip().replace("preset:", "")
            if anim_command.lower() in ['none', '']:
                print(f"[{index+1}/{len(records)}] Skipping {name}: No animation assigned (Anim: none)")
                continue
            
            habit_events = [m for m in user_metrics if m.get('modelName') == name]
            downloads = sum(1 for e in habit_events if e.get('eventType') == 'model_download')
            likes = sum(1 for e in habit_events if e.get('eventType') == 'model_rating' and e.get('rating') == 'like')
            dislikes = sum(1 for e in habit_events if e.get('eventType') == 'model_rating' and e.get('rating') == 'dislike')

            total_obj_ratings = likes + dislikes
            like_ratio = round(likes / total_obj_ratings * 100, 2) if total_obj_ratings > 0 else 0.0
            dislike_ratio = round(dislikes / total_obj_ratings * 100, 2) if total_obj_ratings > 0 else 0.0

            if anim_command.lower() in ui_animations:
                ui_count += 1
                print(f"[{index+1}/{len(records)}] Skipping Blender (UI preset): {name} (UI Anim: {anim_command}) | 👍 {likes} 👎 {dislikes}")
                row = {
                    'habit_name': name, 'anim_command': anim_command, 'is_skeletal': False,
                    'model_downloads': downloads, 'model_likes': likes, 'model_dislikes': dislikes,
                    'model_like_ratio': like_ratio, 'model_dislike_ratio': dislike_ratio,
                    'max_edge_expansion_pct': 0, 'foot_skating_ratio': 0, 'jitter_acceleration': 0,
                    'mesh_clipping_ratio': 0, 'volume_variation_pct': 0, 'edge_stretch_variance': 0,
                    'avg_velocity_l2': 0, 'avg_acceleration_l2': 0, 'jitter_score': 0,
                    'is_watertight_base': False, 'is_watertight_anim': False,
                    'clip_action_score': 0, 'image_reward_score': 0
                }
                results.append(row)
                continue
                
            print(f"[{index+1}/{len(records)}] Analyzing Skeletal animation: {name} | Cmd: {anim_command} | 👍 {likes} 👎 {dislikes}")
            metrics = evaluator.evaluate_glb_with_blender(glb_path)
            
            is_skeletal = metrics.get("valid_rig", False)
            ai_metrics = {"clip_action_score": 0.0, "image_reward_score": 0.0}
            
            if is_skeletal:
                skeletal_count += 1
                render_path = metrics.get("render_path", "")
                ai_metrics = evaluator.run_ai_metrics(render_path, anim_command)
                
                # FULL TERMINAL OUTPUT
                print(f"  -> ✅ Rig valid")
                print(f"     📐 [Geometry] Max expansion: {metrics.get('max_edge_expansion_pct', 0)}% | Edge stretch var: {metrics.get('edge_stretch_variance', 0)} | Volume variation: {metrics.get('volume_variation_pct', 0)}%")
                print(f"     💥 [Physics] Clipping ratio: {metrics.get('mesh_clipping_ratio', 0)} | Foot skating: {metrics.get('foot_skating_ratio', 0)}%")
                print(f"     〰️ [Motion] Jitter accel: {metrics.get('jitter_acceleration', 0)} | Vel L2: {metrics.get('avg_velocity_l2', 0)} | Acc L2: {metrics.get('avg_acceleration_l2', 0)} | Jitter score: {metrics.get('jitter_score', 0)}")
                print(f"     💧 [Structure] Watertight: Base={metrics.get('is_watertight_base', False)} -> Anim={metrics.get('is_watertight_anim', False)}")
                print(f"     🤖 [AI] CLIP score: {ai_metrics.get('clip_action_score', 0)} | ImageReward: {ai_metrics.get('image_reward_score', 0)}")
                print(f"  -> Model downloads: {downloads} | 👍 {likes} 👎 {dislikes} | Like ratio: {like_ratio}% | Dislike ratio: {dislike_ratio}%")
            else:
                ui_count += 1
                print(f"  -> Error: {metrics.get('error_msg', 'Invalid skeleton')}. Routed to UI fallback count.")
                
            row = {
                'habit_name': name, 'anim_command': anim_command, 'is_skeletal': is_skeletal,
                'model_downloads': downloads, 'model_likes': likes, 'model_dislikes': dislikes,
                'model_like_ratio': like_ratio, 'model_dislike_ratio': dislike_ratio,
                'max_edge_expansion_pct': metrics.get("max_edge_expansion_pct", 0),
                'foot_skating_ratio': metrics.get("foot_skating_ratio", 0),
                'jitter_acceleration': metrics.get("jitter_acceleration", 0),
                'mesh_clipping_ratio': metrics.get("mesh_clipping_ratio", 0),
                'volume_variation_pct': metrics.get("volume_variation_pct", 0),
                'edge_stretch_variance': metrics.get("edge_stretch_variance", 0),
                'avg_velocity_l2': metrics.get("avg_velocity_l2", 0),
                'avg_acceleration_l2': metrics.get("avg_acceleration_l2", 0),
                'jitter_score': metrics.get("jitter_score", 0),
                'is_watertight_base': metrics.get("is_watertight_base", False),
                'is_watertight_anim': metrics.get("is_watertight_anim", False),
                **ai_metrics
            }
            results.append(row)
        
        if not results:
            print("No valid animation results to save.")
            return
        
        output_csv = os.path.join(os.path.dirname(input_csv), "anim_results.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)

        if message_metrics and USE_MLFLOW:
            message_metrics_output = "data/message_metrics_artifact_anim.csv"
            os.makedirs(os.path.dirname(message_metrics_output), exist_ok=True)
            
            agg_msgs = {}
            for m in message_metrics:
                text = m.get('message', '').strip()
                if text == '': continue
                entry = agg_msgs.setdefault(text, {'likes':0,'dislikes':0})
                if m.get('eventType') == 'message_rating':
                    if m.get('rating') == 'like': entry['likes'] += 1
                    elif m.get('rating') == 'dislike': entry['dislikes'] += 1
            
            with open(message_metrics_output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['message', 'likes', 'dislikes'])
                writer.writeheader()
                for text, counts in agg_msgs.items():
                    writer.writerow({'message': text, 'likes': counts['likes'], 'dislikes': counts['dislikes']})
            mlflow.log_artifact(message_metrics_output)
        
        total_models = len(results)

        agg = {
            "total_ui_animations": ui_count,
            "total_skeletal_animations": skeletal_count,
            "skeletal_routing_ratio": (skeletal_count / total_models * 100) if total_models > 0 else 0,
            "total_model_downloads": sum(r.get('model_downloads', 0) for r in results),
            "total_model_likes": sum(r.get('model_likes', 0) for r in results),
            "total_model_dislikes": sum(r.get('model_dislikes', 0) for r in results)
        }

        total_model_ratings = agg["total_model_likes"] + agg["total_model_dislikes"]
        
        if total_model_ratings > 0:
            agg["model_like_ratio"] = round(agg["total_model_likes"] / total_model_ratings * 100, 2)
            agg["model_dislike_ratio"] = round(agg["total_model_dislikes"] / total_model_ratings * 100, 2)
        else:
            agg["model_like_ratio"] = 0.0
            agg["model_dislike_ratio"] = 0.0

        agg["model_rating_ratio"] = round(total_model_ratings / total_models * 100, 2) if total_models > 0 else 0.0

        models_with_no_rating = sum(1 for r in results if r.get('model_likes', 0) == 0 and r.get('model_dislikes', 0) == 0)
        agg["model_no_rating_ratio"] = round(models_with_no_rating / total_models * 100, 2) if total_models > 0 else 0.0

        valid_skel = [r for r in results if r.get('is_skeletal', False)]

        if valid_skel:
            agg.update({
                "avg_edge_expansion_pct": mean([r['max_edge_expansion_pct'] for r in valid_skel]),
                "avg_foot_skating_ratio": mean([r['foot_skating_ratio'] for r in valid_skel]),
                "avg_clip_action_score": mean([r['clip_action_score'] for r in valid_skel]),
                "watertight_survival_ratio": (sum(1 for r in valid_skel if r['is_watertight_anim']) / len(valid_skel) * 100)
            })
            if HAS_IMAGE_REWARD: agg["avg_image_reward"] = mean([r['image_reward_score'] for r in valid_skel])
        
        if message_metrics:
            msg_likes = sum(1 for m in message_metrics if m.get('eventType') == 'message_rating' and m.get('rating') == 'like')
            msg_dislikes = sum(1 for m in message_metrics if m.get('eventType') == 'message_rating' and m.get('rating') == 'dislike')
            msg_total = len(message_metrics)
            
            msg_ratio_likes = (msg_likes / msg_total * 100) if msg_total > 0 else 0
            msg_ratio_dislikes = (msg_dislikes / msg_total * 100) if msg_total > 0 else 0
            
            agg["total_message_likes"] = msg_likes
            agg["total_message_dislikes"] = msg_dislikes
            agg["message_like_ratio"] = round(msg_ratio_likes, 2)
            agg["message_dislike_ratio"] = round(msg_ratio_dislikes, 2)
            
        for k, v in agg.items(): log_metric_safe(k, v)
        
        print("\n📊 FINAL AGGREGATION (Animation Metrics):")
        print(f"   Routing ratio: {agg['skeletal_routing_ratio']:.1f}% skeletal vs {100-agg['skeletal_routing_ratio']:.1f}% UI fallback")
        print(f"   Total Downloads: {agg['total_model_downloads']}")
        print(f"   Total Likes: 👍 {agg['total_model_likes']} ({agg['model_like_ratio']}%) | Total Dislikes: 👎 {agg['total_model_dislikes']} ({agg['model_dislike_ratio']}%)")
        if valid_skel:
            print(f"   Average CLIP action score: {agg['avg_clip_action_score']:.4f}")
            print(f"   Watertight survival (pose 0 -> anim): {agg['watertight_survival_ratio']:.1f}% of models remained watertight")
        
        if message_metrics:
            print(f"\n💬 Message Metrics Summary:")
            print(f"   Total Messages Rated: {msg_total}")
            print(f"   👍 Likes: {msg_likes} ({msg_ratio_likes:.1f}%)")
            print(f"   👎 Dislikes: {msg_dislikes} ({msg_ratio_dislikes:.1f}%)")

        print(f"\n📊 Per-Object Metrics Breakdown:")
        for r in results:
            obj_name = r['habit_name']
            obj_down = r.get('model_downloads', 0)
            obj_l = r.get('model_likes', 0)
            obj_d = r.get('model_dislikes', 0)
            obj_pct = r.get('model_like_ratio', 0)
            print(f"     {obj_name}: {obj_down} down, {obj_l} 👍 ({obj_pct:.0f}%), {obj_d} 👎 ({100-obj_pct if (obj_l+obj_d)>0 else 0:.0f}%)")

if __name__ == "__main__":
    try:
        from sync_data import auto_sync
        auto_sync()
    except ImportError: pass
    run_animation_evaluation()