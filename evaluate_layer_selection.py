"""
Phase 3 Evaluation — Prop Layer Selection
==========================================
Evaluates the AI-assisted prop layer selection produced by BrightFactory Cortex.
Only habits created during the Phase 3 evaluation window are processed
(PHASE_3_START / PHASE_3_END constants inside run_layer_evaluation).

Inputs:
  data/eval_list.csv          — habit name, prompt, GLB path (from sync_data.py)
  data/cortex_logs.json       — Cortex create_habit logs (fetched from /api/metrics/cortex)
  data/habit_descriptions.json — habit rule descriptions for description-based CLIP
  data/metrics.json           — user interaction events (from sync_data.py)
  buddyforum/public/assets/models/rob-base.glb — Buddy avatar for HOI evaluation

Outputs:
  data/layer_results.csv    — per-habit layer selection metrics
  data/prop_analysis.json   — prop frequency ranking + per-habit WuP/CLIP breakdown
  MLflow run                — aggregated metrics under 'habit_layer_selection_evaluation'

Metrics:
  Semantic  — CLIP Name→Name, Prompt→Prompt, Desc→Desc; WordNet Wu-Palmer (WuP)
  Diversity — coverage diversity (mean pairwise cosine distance across prop embeddings)
  HOI       — D2O (distance to object), L2P (limb-to-prop), contact rate, bone clearance
  Kinematic — MAD score, jitter score
  Intersection — body intersection rate across 3 animation loops (L1/L2/L3)
  Fallback  — filler prop count, theme-aligned filler count
  Latency   — Cortex selection latency (seconds)
  User      — likes, dislikes

Configuration:
  BLENDER_EXEC        — update to local Blender executable path
  MLFLOW_TRACKING_URI — update to your MLflow server
  USE_MLFLOW          — set False to skip MLflow logging
  USE_BLENDER_EVAL    — set False to skip HOI + kinematic metrics
  LOCAL_SERVER_URL    — override the Buddy Forum server URL for this script
"""
import os
import re
import csv
import json
import subprocess
import tempfile
import unicodedata
import contextlib
from datetime import datetime
from statistics import mean

import requests

import numpy as np
import torch
import clip
import mlflow

try:
    import nltk
    from nltk.corpus import wordnet as wn
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False
    print("NLTK not found. Skipping WordNet metrics.")

USE_MLFLOW = True  # Set to True to enable MLflow logging (requires mlflow package and tracking server)
if USE_MLFLOW: import mlflow

LOCAL_SERVER_URL = 'http://YOUR_SERVER_IP:8080/'  # Set to None to use SERVER_URL from sync_data.py

# --- Blender HOI / Kinematic evaluation ---
USE_BLENDER_EVAL = True   # Set False to skip HOI+kinematic metrics
BLENDER_EXEC = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BUDDY_FORUM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'buddyforum'))
# rob-base.glb is uncompressed (no EXT_meshopt_compression); prefer it for Blender
_rob_base = os.path.join(BUDDY_FORUM_DIR, 'public', 'assets', 'models', 'rob-base.glb')
_rob = os.path.join(BUDDY_FORUM_DIR, 'public', 'assets', 'models', 'rob.glb')
BUDDY_GLB_PATH = _rob_base if os.path.exists(_rob_base) else _rob
PROP_MODELS_DIR = os.path.join(BUDDY_FORUM_DIR, 'generated_models')
BONE_NAME = "DEF-Prop_0"
HAND_BONE_NAMES = ["DEF-hand.L", "DEF-hand.R"]   # used for D2O / L2P / Contact Rate

MLFLOW_TRACKING_URI = "http://YOUR_MLFLOW_SERVER:5000"
EXPERIMENT_NAME = "habit_layer_selection_evaluation"
RUN_PARAMS = {
    "model_provider": "BrightFactory Cortex",
    "task_type": "search_prop_selection",
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


class LayerSelectionEvaluator:
    def __init__(self):
        print("Initializing CLIP model for layer selection evaluation...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        if HAS_WORDNET:
            # NLTK 3.8+ renamed punkt → punkt_tab and averaged_perceptron_tagger → averaged_perceptron_tagger_eng
            for resource in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab',
                             'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
        print("Models Loaded.\n" + "="*40)

    def encode_text(self, text):
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    def clip_text_similarity(self, emb_a, emb_b):
        return float(np.dot(emb_a, emb_b))

    def coverage_diversity(self, embeddings):
        if len(embeddings) < 2:
            return 0.0
        dists = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                dists.append(1.0 - sim)
        return round(mean(dists), 4)

    def extract_head_noun(self, text):
        first_phrase = text.split(',')[0].strip()
        if HAS_WORDNET:
            try:
                tokens = nltk.word_tokenize(first_phrase)
                tagged = nltk.pos_tag(tokens)
                for word, tag in tagged:
                    if tag.startswith('NN') and word.isalpha():
                        return word.lower()
            except Exception:
                pass
        words = [w for w in first_phrase.split() if w.isalpha()]
        return words[-1].lower() if words else first_phrase.lower()

    def extract_key_nouns(self, text, habit_name='', max_nouns=3):
        """Extract multiple meaningful nouns from prompt + habit name for multi-noun WuP matching."""
        nouns = []
        seen = set()

        pt_stop = {'de', 'da', 'do', 'das', 'dos', 'e', 'o', 'a', 'os', 'as', 'em', 'no', 'na',
                   'com', 'por', 'para', 'no', 'na', 'ao', 'ou',
                   # Portuguese prefixes that exist in WordNet with wrong English meanings:
                   'auto',   # PT: automatic/self → EN WordNet: automobile
                   }
        # Generic 3D-prompt words with no discriminating value for prop matching.
        # Includes: shared prefixes (dream/gadget), structural/material descriptors,
        # and adjectives that WordNet knows as nouns (automatic=pistol, stainless=steel
        # alloy) but are used as JJ in 3D generation prompts.
        generic_3d_stop = {
            # Shared prefixes / universal 3D-prompt boilerplate:
            'dream', 'gadget', 'detail', 'lighting', 'studio', 'background',
            'isolated', 'realistic', 'stylized', 'modern', 'single', 'compact', 'small',
            'object', 'render', 'text', 'body', 'high',
            'making', 'chamber', 'assembly', 'indicator', 'output', 'tray',
            # 'machine' appears in every Phase 3 prompt — not discriminating:
            'machine',
            # Material/finish descriptors (WordNet knows these as nouns but used as JJ here):
            'automatic', 'stainless', 'transparent', 'robotic', 'glowing',
            'layered', 'sleek', 'subtle', 'brushed', 'folded', 'handheld',
            # Structural/mechanical parts that are noise across all dream-gadget prompts:
            'arm', 'arms', 'metal', 'steel', 'claw', 'spring', 'barrel', 'button',
            'waistband', 'lights', 'pair', 'feeder', 'led', 'sensor', 'sensor',
            # Number words that NLTK sometimes tags as NN:
            'two', 'one', 'three', 'four', 'five',
            # Hyphen-split prefix artifacts (self-dressing → 'self', etc.):
            'self',
            # Gerunds/verbs used structurally in prompts:
            'showing', 'trapping', 'launching', 'hunting', 'writing', 'dressing',
        }

        def _add_noun(w):
            """Add w to nouns+seen (tracking both raw form and WordNet lemma to avoid singular/plural dups)."""
            lemma = (wn.morphy(w, wn.NOUN) or w) if HAS_WORDNET else w
            if w in seen or lemma in seen:
                return False
            nouns.append(w)
            seen.add(w)
            seen.add(lemma)
            return True

        # 1. Nouns from the habit name (English words only — skip Portuguese stop words)
        for word in habit_name.split():
            w = word.lower().strip('.,!?-:')
            if w.isalpha() and w not in pt_stop and w not in generic_3d_stop and len(w) > 2:
                if HAS_WORDNET and wn.synsets(w, pos=wn.NOUN):
                    _add_noun(w)

        # 2. Nouns from the first 3 comma-phrases of the prompt.
        #    Strip the common "Dream gadget:" prefix (shared across all Phase 3 prompts — not discriminating).
        #    Replace hyphens with spaces so "seal-shaped" → "seal shaped", "thesis-writing" → "thesis writing".
        clean_text = re.sub(r'^Dream gadget:\s*', '', text.strip(), flags=re.IGNORECASE)
        clean_text = clean_text.replace('-', ' ')
        first_phrases = ','.join(clean_text.split(',')[:3])

        if HAS_WORDNET:
            candidate_words = []
            try:
                tokens = nltk.word_tokenize(first_phrases)
                tagged = nltk.pos_tag(tokens)
                candidate_words = [word for word, tag in tagged if tag.startswith('NN')]
            except Exception:
                # punkt / tagger not available — scan all words as candidates
                candidate_words = re.split(r'[\s]+', first_phrases)

            for word in candidate_words:
                w = word.lower().strip('.,!?-:')
                if (w.isalpha() and w not in pt_stop and w not in generic_3d_stop and len(w) > 2):
                    if wn.synsets(w, pos=wn.NOUN):
                        # Also reject if the WordNet lemma is in the stop list
                        # (catches plurals/inflections of stopped words, e.g. 'leds' → 'led')
                        if (wn.morphy(w, wn.NOUN) or w) not in generic_3d_stop:
                            _add_noun(w)
                            if len(nouns) >= max_nouns:
                                break

            # Second pass: if POS tagger found fewer than 3 nouns (e.g. tagger tagged
            # a noun as JJ in compound like "layered sandwich assembly"), scan all words
            # in the full clean_text as candidates — POS-tag-free but still WordNet-filtered.
            if len(nouns) < 3:
                for word in re.split(r'[\s,]+', clean_text):
                    w = word.lower().strip('.,!?-:')
                    if (w.isalpha() and w not in pt_stop and w not in generic_3d_stop and len(w) > 2):
                        if wn.synsets(w, pos=wn.NOUN):
                            if (wn.morphy(w, wn.NOUN) or w) not in generic_3d_stop:
                                _add_noun(w)
                                if len(nouns) >= max_nouns:
                                    break

        if not nouns:
            nouns = [self.extract_head_noun(text)]
        return nouns[:max_nouns]

    def wu_palmer_multi(self, target_nouns, prop_nouns):
        """Best WuP score across all target_noun × prop_noun pairs (best-vs-best).
        Returns (score, matched_target_noun, matched_prop_noun)."""
        best_score = None
        best_target = None
        best_prop = None
        for tn in target_nouns:
            for pn in prop_nouns:
                score = self.wu_palmer(tn, pn)
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_target = tn
                    best_prop = pn
        return best_score, best_target, best_prop

    def wu_palmer(self, noun_a, noun_b):
        if not HAS_WORDNET:
            return None
        synsets_a = wn.synsets(noun_a, pos=wn.NOUN)
        synsets_b = wn.synsets(noun_b, pos=wn.NOUN)
        if not synsets_a or not synsets_b:
            return None
        best = 0.0
        for s1 in synsets_a:
            for s2 in synsets_b:
                score = s1.wup_similarity(s2)
                if score and score > best:
                    best = score
        return round(best, 4) if best > 0 else None


def _strip_accents(text):
    """Remove diacritics so 'Água' → 'agua', 'Cão' → 'cao'."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_glb_name(name):
    """Normalize a habit/prompt name to match the GLB filename convention."""
    name = _strip_accents(name.strip().lower())
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')

def resolve_prop_glbs(search_props, habits_prompt_lookup, models_dir=PROP_MODELS_DIR):
    """Map a list of habit names to their GLB file paths in models_dir.

    Tries: normalized habit name, then first word/noun of object_prompt.
    Returns list of resolved (existing) GLB paths (same order as search_props, gaps omitted).
    """
    if not os.path.isdir(models_dir):
        return []
    available = {os.path.splitext(f)[0].lower(): os.path.join(models_dir, f)
                 for f in os.listdir(models_dir) if f.lower().endswith('.glb')}
    resolved = []
    for prop_name in search_props:
        norm = normalize_glb_name(prop_name)
        if norm in available:
            resolved.append(available[norm])
            continue
        # Fallback: first word of the object_prompt
        prompt = habits_prompt_lookup.get(prop_name, '')
        if prompt:
            first_word = normalize_glb_name(prompt.split(',')[0].split()[0]) if prompt.split() else ''
            if first_word and first_word in available:
                resolved.append(available[first_word])
    return resolved

def decompress_buddy_glb(glb_path):
    """Return a meshopt-free copy of glb_path for Blender import.

    gltf-transform reads the compressed GLB (decompressing buffers) and writes
    it back without re-applying compression, producing a plain-binary GLB that
    Blender's importer can handle.  Cached in the 3d-evaluation script directory
    (not inside buddyforum) as <basename>-blender.glb.
    """
    stem = os.path.splitext(os.path.basename(glb_path))[0]
    out_path = os.path.join(SCRIPT_DIR, stem + '-blender.glb')
    if os.path.exists(out_path):
        return out_path

    try:
        # shell=True so npx.cmd is resolved on Windows
        cmd = (
            f'npx --yes --quiet @gltf-transform/cli copy '
            f'"{glb_path}" "{out_path}"'
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        if os.path.exists(out_path):
            print(f"   Buddy GLB decompressed → {os.path.basename(out_path)}")
            return out_path
        print(f"   ⚠️  gltf-transform copy failed (exit {result.returncode}): {result.stderr.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        print("   ⚠️  gltf-transform timed out after 180s")
    except Exception as e:
        print(f"   ⚠️  gltf-transform error: {e}")

    print("   💡 To enable Blender HOI metrics, install gltf-transform once:")
    print("      npm install -g @gltf-transform/cli")
    return glb_path  # caller will receive meshopt error from Blender


def run_blender_prop_eval(prop_glb_paths):
    """Run blender_prop_eval.py headlessly; return the parsed result dict."""
    blender_script = os.path.join(SCRIPT_DIR, "blender_prop_eval.py")
    if not os.path.exists(BUDDY_GLB_PATH):
        return {"error": f"Buddy GLB not found at {BUDDY_GLB_PATH}"}
    if not prop_glb_paths:
        return {"error": "No prop GLBs resolved for this habit"}
    if not os.path.exists(blender_script):
        return {"error": f"blender_prop_eval.py not found at {blender_script}"}

    buddy_glb = decompress_buddy_glb(os.path.abspath(BUDDY_GLB_PATH))

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
        output_json = tmp.name

    try:
        cmd = [
            BLENDER_EXEC, "--background", "--python", blender_script, "--",
            buddy_glb,
            json.dumps([os.path.abspath(p) for p in prop_glb_paths]),
            BONE_NAME,
            output_json,
            json.dumps(HAND_BONE_NAMES),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False, timeout=300, text=True)
        if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
            with open(output_json, 'r') as f:
                return json.load(f)
        # Blender crashed or wrote nothing — surface the last stderr lines
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:]) if proc.stderr else ""
        return {"error": f"Blender produced no output JSON (exit {proc.returncode}). Stderr:\n{stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"error": "Blender timeout (300s)"}
    except FileNotFoundError:
        return {"error": f"Blender executable not found: '{BLENDER_EXEC}'. Update BLENDER_EXEC."}
    finally:
        if os.path.exists(output_json):
            os.unlink(output_json)


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

def load_cortex_logs(cortex_logs_path="data/cortex_logs.json"):
    """Load create_habit Cortex entries fetched from /api/metrics/cortex, keyed by habit_name."""
    cortex_map = {}
    if not os.path.exists(cortex_logs_path):
        print(f"⚠️  {cortex_logs_path} not found. Run auto-sync to fetch Cortex logs.")
        return cortex_map
    with open(cortex_logs_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return cortex_map
    for entry in data:
        habit_name = entry.get('habit_name', '').strip()
        if habit_name:
            cortex_map[habit_name] = {
                'search_props': entry.get('search_props', []),
                'latency': entry.get('latency_s', -1.0),
                'object_prompt': entry.get('object_prompt', ''),
            }
    return cortex_map


RELATED_WUP_THRESHOLD  = 0.65   # WuP ≥ this → semantically related
RELATED_CLIP_THRESHOLD = 0.55   # CLIP prompt sim ≥ this → semantically related


def run_layer_evaluation(input_csv="data/eval_list.csv", output_csv="data/layer_results.csv"):
    if not os.path.exists(input_csv): return
    records = []
    with open(input_csv, mode='r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    habits_prompt_lookup = {r['Nome']: r.get('Prompt', '') for r in records}

    habit_desc_lookup = {}
    desc_path = "data/habit_descriptions.json"
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            try: habit_desc_lookup = json.load(f)
            except: pass

    user_metrics = load_user_metrics()
    message_metrics = load_message_metrics()
    cortex_map = load_cortex_logs()

    if user_metrics:
        sample = user_metrics[0]
        if not {'modelName', 'eventType'}.issubset(sample.keys()):
            print("⚠️  metrics.json schema is Cortex logs (not habit events). Likes/downloads will be 0.")

    evaluator = LayerSelectionEvaluator()
    results = []

    if USE_MLFLOW: mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); mlflow.set_experiment(EXPERIMENT_NAME)

    with optional_mlflow_run("Layer_Eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")):
        PHASE_3_START = datetime.fromisoformat("2026-05-26T00:00:00")
        PHASE_3_END   = datetime.fromisoformat("2026-05-29T00:00:00")  # exclusive — only the 6 surveyed habits

        # Pre-collect Phase 3 habit names — since Phase 3 is a single themed week,
        # this set IS the weekly theme pool (no external lookup needed).
        phase3_habit_names = set()
        for record in records:
            created_at_str = record.get('Data', '')
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00').replace('+00:00', ''))
                if PHASE_3_START <= created_at < PHASE_3_END:
                    phase3_habit_names.add(record.get('Nome', '').strip())
            except (ValueError, AttributeError):
                pass
        print(f"📅 Phase 3 habit pool ({len(phase3_habit_names)} habits = weekly theme): {sorted(phase3_habit_names)}\n")

        for index, record in enumerate(records):
            name = record.get('Nome', f'Habit_{index}')

            # Skip habits created before Phase 3 (prop layer selection) was deployed
            created_at_str = record.get('Data', '')
            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00').replace('+00:00', ''))
                if created_at < PHASE_3_START or created_at >= PHASE_3_END:
                    print(f"[{index+1}/{len(records)}] Skipping {name}: outside Phase 3 window ({created_at.date()}).")
                    continue
            except (ValueError, AttributeError):
                pass  # If date is missing or unparseable, include the habit anyway

            if name not in cortex_map:
                print(f"[{index+1}/{len(records)}] Skipping {name}: no Cortex create_habit log found.")
                continue

            cortex_entry = cortex_map[name]
            search_props = cortex_entry['search_props']
            latency = cortex_entry['latency']
            object_prompt = cortex_entry.get('object_prompt') or record.get('Prompt', '')

            habit_events = [m for m in user_metrics if m.get('modelName') == name]
            downloads = sum(1 for e in habit_events if e.get('eventType') == 'model_download')
            likes = sum(1 for e in habit_events if e.get('eventType') == 'model_rating' and e.get('rating') == 'like')
            dislikes = sum(1 for e in habit_events if e.get('eventType') == 'model_rating' and e.get('rating') == 'dislike')
            total_ratings = likes + dislikes
            like_ratio = round(likes / total_ratings * 100, 2) if total_ratings > 0 else 0.0
            dislike_ratio = round(dislikes / total_ratings * 100, 2) if total_ratings > 0 else 0.0

            prop_count = len(search_props)
            rule_compliant = int(5 <= prop_count <= 10)

            # Prompt-to-prompt CLIP (visual similarity)
            target_prompt_emb = evaluator.encode_text(object_prompt) if object_prompt else None
            prop_embs = []
            clip_prompt_sims = []
            for prop_name in search_props:
                prop_prompt = habits_prompt_lookup.get(prop_name, prop_name)
                prop_emb = evaluator.encode_text(prop_prompt)
                prop_embs.append(prop_emb)
                if target_prompt_emb is not None:
                    clip_prompt_sims.append(evaluator.clip_text_similarity(target_prompt_emb, prop_emb))

            clip_mean = round(mean(clip_prompt_sims), 4) if clip_prompt_sims else 0.0
            clip_min = round(min(clip_prompt_sims), 4) if clip_prompt_sims else 0.0

            # Description-to-description CLIP (contextual/behavioural similarity)
            target_desc = habit_desc_lookup.get(name, '')
            target_desc_emb = evaluator.encode_text(target_desc) if target_desc else None
            clip_desc_sims = []
            for prop_name in search_props:
                prop_desc = habit_desc_lookup.get(prop_name, prop_name)
                prop_desc_emb = evaluator.encode_text(prop_desc)
                if target_desc_emb is not None:
                    clip_desc_sims.append(evaluator.clip_text_similarity(target_desc_emb, prop_desc_emb))

            clip_desc_mean = round(mean(clip_desc_sims), 4) if clip_desc_sims else -1.0
            clip_desc_min = round(min(clip_desc_sims), 4) if clip_desc_sims else -1.0

            # Name-to-name CLIP (label-level similarity)
            target_name_emb = evaluator.encode_text(name)
            clip_name_sims = []
            for prop_name in search_props:
                prop_name_emb = evaluator.encode_text(prop_name)
                clip_name_sims.append(evaluator.clip_text_similarity(target_name_emb, prop_name_emb))

            clip_name_mean = round(mean(clip_name_sims), 4) if clip_name_sims else 0.0
            clip_name_min = round(min(clip_name_sims), 4) if clip_name_sims else 0.0

            diversity = evaluator.coverage_diversity(prop_embs)

            wup_scores = []
            wup_details = []  # per-prop breakdown: prop name, extracted nouns, scores, fallback classification
            target_nouns = []
            if HAS_WORDNET and object_prompt:
                target_nouns = evaluator.extract_key_nouns(object_prompt, habit_name=name)
            for i, prop_name in enumerate(search_props):
                prop_prompt = habits_prompt_lookup.get(prop_name, prop_name)
                # Multi-noun extraction for props mirrors what we do for the target habit.
                # Falls back to extract_head_noun only when extract_key_nouns returns nothing.
                prop_nouns = evaluator.extract_key_nouns(prop_prompt, habit_name=prop_name)
                if not prop_nouns:
                    prop_nouns = [evaluator.extract_head_noun(prop_prompt)]
                wup_score, matched_target_noun, matched_prop_noun = (
                    evaluator.wu_palmer_multi(target_nouns, prop_nouns)
                    if target_nouns else (None, None, None)
                )
                clip_score = clip_prompt_sims[i] if i < len(clip_prompt_sims) else None

                # Fallback classification — additive flags (not mutually exclusive):
                #   is_related:      CLIP Prompt→Prompt ≥ threshold (visual-semantic alignment).
                #                    WuP is computed and reported as a complementary lexical-
                #                    semantic metric (per professor's requirement) but does NOT
                #                    drive this binary classification — noun extraction from 3D
                #                    visual prompts introduces too much structural noise for WuP
                #                    to be a reliable classifier here.
                #   is_theme_member: prop is a Phase 3 habit (same weekly theme)
                # A prop can be both related AND theme, or a filler with/without theme membership.
                # "Filler" = NOT related; a filler that is theme_member followed the Ruler's
                # fallback rule correctly; a filler that is NOT theme_member violated it.
                is_related = (clip_score is not None and clip_score >= RELATED_CLIP_THRESHOLD)
                is_theme_member = prop_name in phase3_habit_names
                is_filler = not is_related
                is_theme_aligned_filler = is_filler and is_theme_member

                wup_details.append({
                    'prop': prop_name,
                    'prop_nouns': prop_nouns,
                    'matched_prop_noun': matched_prop_noun,
                    'matched_target_noun': matched_target_noun,
                    'wup': round(wup_score, 4) if wup_score is not None else None,
                    'clip_prompt': round(clip_score, 4) if clip_score is not None else None,
                    'is_related': is_related,
                    'is_theme_member': is_theme_member,
                    'is_filler': is_filler,
                    'is_theme_aligned_filler': is_theme_aligned_filler,
                })
                if wup_score is not None:
                    wup_scores.append(wup_score)

            wup_mean = round(mean(wup_scores), 4) if wup_scores else -1.0
            wup_min = round(min(wup_scores), 4) if wup_scores else -1.0

            # Best / worst prop by WuP and by CLIP
            valid_wup_details = [d for d in wup_details if d['wup'] is not None]
            best_wup_prop  = max(valid_wup_details, key=lambda d: d['wup'],  default={}).get('prop', '')
            worst_wup_prop = min(valid_wup_details, key=lambda d: d['wup'],  default={}).get('prop', '')
            valid_clip_details = [d for d in wup_details if d['clip_prompt'] is not None]
            best_clip_prop  = max(valid_clip_details, key=lambda d: d['clip_prompt'], default={}).get('prop', '')
            worst_clip_prop = min(valid_clip_details, key=lambda d: d['clip_prompt'], default={}).get('prop', '')

            # Fallback / theme membership summary (additive flags)
            related_prop_count          = sum(1 for d in wup_details if d['is_related'])
            theme_member_count          = sum(1 for d in wup_details if d['is_theme_member'])
            filler_prop_count           = sum(1 for d in wup_details if d['is_filler'])
            theme_aligned_filler_count  = sum(1 for d in wup_details if d['is_theme_aligned_filler'])
            non_theme_filler_count      = sum(1 for d in wup_details if d['is_filler'] and not d['is_theme_member'])
            filler_theme_alignment_rate = (
                round(theme_aligned_filler_count / filler_prop_count * 100, 1)
                if filler_prop_count > 0 else 100.0
            )

            # HOI + Kinematic metrics via Blender
            mad_score = jitter_score = d2o_mean = d2o_min = l2p_mean = l2p_min = contact_rate = bone_to_avatar_min = bone_to_avatar_mean = body_intersect_rate = body_intersect_loop1 = body_intersect_loop2 = body_intersect_loop3 = -1.0
            if USE_BLENDER_EVAL:
                prop_glbs = resolve_prop_glbs(search_props, habits_prompt_lookup)
                if prop_glbs:
                    blender_result = run_blender_prop_eval(prop_glbs)
                    if "error" in blender_result:
                        print(f"  ⚠️  Blender eval: {blender_result['error']}")
                    else:
                        kin = blender_result.get("kinematic", {})
                        mad_score = kin.get("mad_score", -1.0)
                        jitter_score = kin.get("jitter_score", -1.0)
                        valid_hoi = [p for p in blender_result.get("props", []) if "error" not in p]
                        if valid_hoi:
                            def _avg(key): v=[p[key] for p in valid_hoi if p.get(key,-1)>=0]; return round(mean(v),4) if v else -1.0
                            d2o_mean   = _avg("d2o_mean")
                            d2o_min    = _avg("d2o_min")
                            l2p_mean   = _avg("l2p_mean")
                            l2p_min    = _avg("l2p_min")
                            contact_rate         = _avg("contact_rate")
                            bone_to_avatar_min   = _avg("bone_to_avatar_min")
                            bone_to_avatar_mean  = _avg("bone_to_avatar_mean")
                            body_intersect_rate   = _avg("body_intersect_rate")
                            body_intersect_loop1  = _avg("body_intersect_loop1")
                            body_intersect_loop2  = _avg("body_intersect_loop2")
                            body_intersect_loop3  = _avg("body_intersect_loop3")
                else:
                    print(f"  ⚠️  Blender eval skipped: no GLBs resolved for props {search_props[:3]}")

            print(f"[{index+1}/{len(records)}] {name} | Props: {prop_count} | Compliant: {'✅' if rule_compliant else '❌'} | 👍 {likes} 👎 {dislikes}")
            print(f"  🔍 CLIP Prompt→Prompt: mean={clip_mean:.4f} min={clip_min:.4f} | Diversity: {diversity:.4f}")
            if clip_desc_mean >= 0:
                print(f"  📝 CLIP Desc→Desc:   mean={clip_desc_mean:.4f} min={clip_desc_min:.4f}")
            print(f"  🏷️  CLIP Name→Name:   mean={clip_name_mean:.4f} min={clip_name_min:.4f}")
            if HAS_WORDNET:
                print(f"  📖 WordNet WuP: mean={wup_mean:.4f} min={wup_min:.4f} | target nouns: {target_nouns}")
            for d in wup_details:
                best_marker = ' 🏆' if d['prop'] == best_wup_prop else ('  ⚠️' if d['prop'] == worst_wup_prop else '    ')
                wup_str  = f"{d['wup']:.4f}"  if d['wup']  is not None else "  N/A"
                clip_str = f"{d['clip_prompt']:.4f}" if d['clip_prompt'] is not None else "  N/A"
                # Show the matched noun pair (prop↔target) when WuP found a match,
                # or just the first extracted prop noun when no WuP match exists.
                if d.get('matched_prop_noun') and d.get('matched_target_noun'):
                    match_str = f" '{d['matched_prop_noun']}'↔'{d['matched_target_noun']}'"
                else:
                    first_pn = d['prop_nouns'][0] if d.get('prop_nouns') else '?'
                    match_str = f" '{first_pn}'"
                # Tag shows: relatedness + theme membership (additive)
                if d['is_related'] and d['is_theme_member']:
                    tag = ' [related+theme✅]'
                elif d['is_related']:
                    tag = ' [related]'
                elif d['is_theme_member']:
                    tag = ' [filler/theme✅]'
                else:
                    tag = ' [filler/theme❌]'
                print(f"      {best_marker} {d['prop']}:{match_str} | wup={wup_str} | clip={clip_str}{tag}")
            print(f"  🎯 Fallback: {related_prop_count} related ({theme_member_count} theme) | "
                  f"{filler_prop_count} fillers, {theme_aligned_filler_count} theme-aligned")
            if mad_score >= 0:
                print(f"  🦾 Kinematic: MAD={mad_score:.4f} Jitter={jitter_score:.4f}")
            if d2o_mean >= 0:
                loop_str = ""
                if body_intersect_loop1 >= 0:
                    loop_str = f" [L1={body_intersect_loop1:.0f}% L2={body_intersect_loop2:.0f}% L3={body_intersect_loop3:.0f}%]"
                intersect_str = f" | BodyIntersect={body_intersect_rate:.1f}%{loop_str}" if body_intersect_rate >= 0 else ""
                print(f"  📐 HOI: D2O mean={d2o_mean:.4f} min={d2o_min:.4f} | L2P mean={l2p_mean:.4f} min={l2p_min:.4f} | Contact={contact_rate:.1f}% | BoneClr={bone_to_avatar_min:.4f}m{intersect_str}")
            print(f"  ⏱️  Cortex Latency: {latency:.3f}s")

            row = {
                'habit_name': name,
                'object_prompt': object_prompt,
                'search_props': json.dumps(search_props, ensure_ascii=False),
                'prop_count': prop_count,
                'rule_compliant': rule_compliant,
                'clip_name_mean': clip_name_mean,
                'clip_name_min': clip_name_min,
                'clip_prompt_mean': clip_mean,
                'clip_prompt_min': clip_min,
                'clip_desc_mean': clip_desc_mean,
                'clip_desc_min': clip_desc_min,
                'coverage_diversity': diversity,
                'wordnet_wup_mean': wup_mean,
                'wordnet_wup_min': wup_min,
                'target_nouns': json.dumps(target_nouns, ensure_ascii=False),
                'best_wup_prop': best_wup_prop,
                'worst_wup_prop': worst_wup_prop,
                'best_clip_prop': best_clip_prop,
                'worst_clip_prop': worst_clip_prop,
                'related_prop_count': related_prop_count,
                'theme_member_count': theme_member_count,
                'filler_prop_count': filler_prop_count,
                'theme_aligned_filler_count': theme_aligned_filler_count,
                'non_theme_filler_count': non_theme_filler_count,
                'filler_theme_alignment_rate': filler_theme_alignment_rate,
                'wup_details': json.dumps(wup_details, ensure_ascii=False),
                'cortex_latency_s': round(latency, 4) if latency >= 0 else -1.0,
                'mad_score': mad_score,
                'jitter_score': jitter_score,
                'd2o_mean': d2o_mean,
                'd2o_min': d2o_min,
                'l2p_mean': l2p_mean,
                'l2p_min': l2p_min,
                'contact_rate': contact_rate,
                'bone_to_avatar_min': bone_to_avatar_min,
                'bone_to_avatar_mean': bone_to_avatar_mean,
                'body_intersect_rate': body_intersect_rate,
                'body_intersect_loop1': body_intersect_loop1,
                'body_intersect_loop2': body_intersect_loop2,
                'body_intersect_loop3': body_intersect_loop3,
                'model_downloads': downloads,
                'model_likes': likes,
                'model_dislikes': dislikes,
                'model_like_ratio': like_ratio,
                'model_dislike_ratio': dislike_ratio,
            }
            results.append(row)

        if not results:
            print("No Phase 3 habits found in Cortex logs.")
            return

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)

        # --- Prop frequency & per-habit best/worst analysis ---
        from collections import Counter
        all_props_flat = []
        for r in results:
            all_props_flat.extend(json.loads(r['search_props']))
        prop_freq = Counter(all_props_flat).most_common()

        prop_analysis = {
            'phase3_habit_pool': sorted(phase3_habit_names),
            'thresholds': {
                'related_wup': RELATED_WUP_THRESHOLD,
                'related_clip': RELATED_CLIP_THRESHOLD,
            },
            'prop_frequency': [{'prop': p, 'count': c} for p, c in prop_freq],
            'per_habit': [
                {
                    'habit': r['habit_name'],
                    'target_nouns': json.loads(r.get('target_nouns', '[]')),
                    'related_prop_count': r.get('related_prop_count', 0),
                    'filler_prop_count': r.get('filler_prop_count', 0),
                    'theme_aligned_filler_count': r.get('theme_aligned_filler_count', 0),
                    'filler_theme_alignment_rate': r.get('filler_theme_alignment_rate', 100.0),
                    'best_wup_prop': r.get('best_wup_prop', ''),
                    'worst_wup_prop': r.get('worst_wup_prop', ''),
                    'best_clip_prop': r.get('best_clip_prop', ''),
                    'worst_clip_prop': r.get('worst_clip_prop', ''),
                    'wup_details': json.loads(r.get('wup_details', '[]')),
                }
                for r in results
            ]
        }
        prop_analysis_path = "data/prop_analysis.json"
        os.makedirs(os.path.dirname(prop_analysis_path), exist_ok=True)
        with open(prop_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(prop_analysis, f, ensure_ascii=False, indent=2)
        print(f"\n📦 Prop analysis saved to {prop_analysis_path}")
        if USE_MLFLOW:
            mlflow.log_artifact(prop_analysis_path)

        if message_metrics and USE_MLFLOW:
            message_metrics_output = "data/message_metrics_artifact_layer.csv"
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
            "total_habits_evaluated": total_models,
            "rule_compliance_rate": round(sum(r['rule_compliant'] for r in results) / total_models * 100, 2),
            "avg_prop_count": round(mean([r['prop_count'] for r in results]), 2),
            "avg_clip_name_mean": round(mean([r['clip_name_mean'] for r in results]), 4),
            "avg_clip_name_min": round(mean([r['clip_name_min'] for r in results]), 4),
            "avg_clip_prompt_mean": round(mean([r['clip_prompt_mean'] for r in results]), 4),
            "avg_clip_prompt_min": round(mean([r['clip_prompt_min'] for r in results]), 4),
            "avg_coverage_diversity": round(mean([r['coverage_diversity'] for r in results]), 4),
            "total_model_downloads": sum(r['model_downloads'] for r in results),
            "total_model_likes": sum(r['model_likes'] for r in results),
            "total_model_dislikes": sum(r['model_dislikes'] for r in results),
        }

        valid_latencies = [r['cortex_latency_s'] for r in results if r['cortex_latency_s'] >= 0]
        if valid_latencies:
            agg["avg_cortex_latency_s"] = round(mean(valid_latencies), 4)

        for key, col in [("avg_mad_score", "mad_score"), ("avg_jitter_score", "jitter_score"),
                         ("avg_d2o_mean", "d2o_mean"), ("avg_d2o_min", "d2o_min"),
                         ("avg_l2p_mean", "l2p_mean"), ("avg_l2p_min", "l2p_min"),
                         ("avg_contact_rate", "contact_rate"),
                         ("avg_bone_to_avatar_min", "bone_to_avatar_min"),
                         ("avg_bone_to_avatar_mean", "bone_to_avatar_mean"),
                         ("avg_body_intersect_rate", "body_intersect_rate"),
                         ("avg_body_intersect_loop1", "body_intersect_loop1"),
                         ("avg_body_intersect_loop2", "body_intersect_loop2"),
                         ("avg_body_intersect_loop3", "body_intersect_loop3")]:
            vals = [r[col] for r in results if r[col] >= 0]
            if vals:
                agg[key] = round(mean(vals), 4)

        desc_valid = [r['clip_desc_mean'] for r in results if r['clip_desc_mean'] >= 0]
        if desc_valid:
            agg["avg_clip_desc_mean"] = round(mean(desc_valid), 4)
            agg["avg_clip_desc_min"] = round(mean([r['clip_desc_min'] for r in results if r['clip_desc_min'] >= 0]), 4)

        wup_valid = [r['wordnet_wup_mean'] for r in results if r['wordnet_wup_mean'] >= 0]
        if wup_valid:
            agg["avg_wordnet_wup_mean"] = round(mean(wup_valid), 4)

        # Fallback / weekly theme alignment aggregates
        agg["avg_related_prop_count"]         = round(mean([r['related_prop_count'] for r in results]), 2)
        agg["avg_theme_member_count"]         = round(mean([r['theme_member_count'] for r in results]), 2)
        agg["avg_filler_prop_count"]          = round(mean([r['filler_prop_count'] for r in results]), 2)
        agg["avg_theme_aligned_filler_count"] = round(mean([r['theme_aligned_filler_count'] for r in results]), 2)
        agg["avg_non_theme_filler_count"]     = round(mean([r['non_theme_filler_count'] for r in results]), 2)
        filled_habits = [r for r in results if r['filler_prop_count'] > 0]
        agg["avg_filler_theme_alignment_rate"] = (
            round(mean([r['filler_theme_alignment_rate'] for r in filled_habits]), 1)
            if filled_habits else 100.0
        )
        agg["habits_needing_fallback"] = len(filled_habits)

        total_model_ratings = agg["total_model_likes"] + agg["total_model_dislikes"]
        if total_model_ratings > 0:
            agg["model_like_ratio"] = round(agg["total_model_likes"] / total_model_ratings * 100, 2)
            agg["model_dislike_ratio"] = round(agg["total_model_dislikes"] / total_model_ratings * 100, 2)
        else:
            agg["model_like_ratio"] = 0.0
            agg["model_dislike_ratio"] = 0.0

        agg["model_rating_ratio"] = round(total_model_ratings / total_models * 100, 2) if total_models > 0 else 0.0
        models_with_no_rating = sum(1 for r in results if r['model_likes'] == 0 and r['model_dislikes'] == 0)
        agg["model_no_rating_ratio"] = round(models_with_no_rating / total_models * 100, 2) if total_models > 0 else 0.0

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
        if USE_MLFLOW: mlflow.log_artifact(output_csv)

        print(f"\n📊 FINAL AGGREGATION (Layer Selection Metrics):")
        print(f"   Habits evaluated: {total_models}")
        print(f"   ✅ Rule compliance rate: {agg['rule_compliance_rate']}% ({sum(r['rule_compliant'] for r in results)}/{total_models} compliant)")
        print(f"   📐 Avg prop count: {agg['avg_prop_count']:.1f}")
        print(f"   🏷️  Avg CLIP Name→Name:   mean={agg['avg_clip_name_mean']:.4f} | min={agg['avg_clip_name_min']:.4f}")
        print(f"   🔍 Avg CLIP Prompt→Prompt: mean={agg['avg_clip_prompt_mean']:.4f} | min={agg['avg_clip_prompt_min']:.4f}")
        if 'avg_clip_desc_mean' in agg:
            print(f"   📝 Avg CLIP Desc→Desc:   mean={agg['avg_clip_desc_mean']:.4f} | min={agg['avg_clip_desc_min']:.4f}")
        print(f"   🌐 Avg coverage diversity: {agg['avg_coverage_diversity']:.4f}")
        if 'avg_wordnet_wup_mean' in agg:
            print(f"   📖 Avg WordNet Wu-Palmer: {agg['avg_wordnet_wup_mean']:.4f}")
        print(f"   🎯 Fallback (avg per habit): {agg['avg_related_prop_count']:.1f} related "
              f"({agg['avg_theme_member_count']:.1f} theme) | "
              f"{agg['avg_filler_prop_count']:.1f} fillers, "
              f"{agg['avg_theme_aligned_filler_count']:.1f} theme-aligned "
              f"({agg['habits_needing_fallback']}/{total_models} habits used fallback)")
        if 'avg_cortex_latency_s' in agg:
            print(f"   ⏱️  Avg Cortex latency: {agg['avg_cortex_latency_s']:.3f}s")
        if 'avg_mad_score' in agg:
            print(f"   🦾 Avg Kinematic: MAD={agg['avg_mad_score']:.4f} | Jitter={agg.get('avg_jitter_score', -1):.4f}")
        if 'avg_d2o_mean' in agg:
            print(f"   📐 Avg HOI: D2O mean={agg['avg_d2o_mean']:.4f} min={agg.get('avg_d2o_min',-1):.4f} | "
                  f"L2P mean={agg['avg_l2p_mean']:.4f} min={agg.get('avg_l2p_min',-1):.4f} | "
                  f"Contact={agg.get('avg_contact_rate',-1):.1f}% | "
                  f"BoneClr={agg.get('avg_bone_to_avatar_min',-1):.4f}m | "
                  f"BodyIntersect={agg.get('avg_body_intersect_rate',-1):.1f}%")
        if prop_freq:
            print(f"\n   🔁 Prop frequency ranking (most used in search pool):")
            for prop, count in prop_freq:
                print(f"      {count}x  {prop}")

        print(f"   Total Downloads: {agg['total_model_downloads']}")
        print(f"   Total Likes: 👍 {agg['total_model_likes']} ({agg['model_like_ratio']}%) | Total Dislikes: 👎 {agg['total_model_dislikes']} ({agg['model_dislike_ratio']}%)")

        if message_metrics:
            print(f"\n💬 Message Metrics Summary:")
            print(f"   Total Messages Rated: {msg_total}")
            print(f"   👍 Likes: {msg_likes} ({msg_ratio_likes:.1f}%)")
            print(f"   👎 Dislikes: {msg_dislikes} ({msg_ratio_dislikes:.1f}%)")

        print(f"\n📊 Per-Object Metrics Breakdown:")
        for r in results:
            obj_l = r.get('model_likes', 0)
            obj_d = r.get('model_dislikes', 0)
            obj_pct = r.get('model_like_ratio', 0)
            clip_desc_str = f", desc={r['clip_desc_mean']:.3f}" if r['clip_desc_mean'] >= 0 else ""
            hoi_str = (f", d2o={r['d2o_mean']:.3f}, min={r['d2o_min']:.3f}, cont={r['contact_rate']:.0f}%, clr={r['bone_to_avatar_min']:.3f}m"
                       if r['d2o_mean'] >= 0 else "")
            kin_str = f", mad={r['mad_score']:.3f}" if r['mad_score'] >= 0 else ""
            print(f"     {r['habit_name']}: {r['prop_count']} props, name={r['clip_name_mean']:.3f}, prompt={r['clip_prompt_mean']:.3f}{clip_desc_str}, div={r['coverage_diversity']:.3f}{kin_str}{hoi_str}, lat={r['cortex_latency_s']:.2f}s | 👍 {obj_l} 👎 {obj_d} ({obj_pct:.0f}%)")


if __name__ == "__main__":
    if USE_MLFLOW:
        print(f"🔍 MLflow Version: {mlflow.__version__}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n--- Running Auto-Sync ---")
    try:
        import sync_data
        server_url = LOCAL_SERVER_URL or sync_data.SERVER_URL
        if LOCAL_SERVER_URL:
            sync_data.SERVER_URL = LOCAL_SERVER_URL
        print(f"   Using server: {server_url}")
        sync_data.auto_sync()
        # Fetch Cortex logs from the dedicated endpoint
        cortex_response = requests.get(f"{server_url}/api/metrics/cortex")
        cortex_response.raise_for_status()
        cortex_data = cortex_response.json()
        cortex_log_path = "data/cortex_logs.json"
        os.makedirs("data", exist_ok=True)
        if len(cortex_data) == 0 and os.path.exists(cortex_log_path):
            # Server returned empty — keep the cached file (e.g. running locally without Cortex)
            existing_count = len(json.load(open(cortex_log_path, encoding='utf-8')))
            print(f"⚠️  Server returned 0 Cortex entries — keeping cached {cortex_log_path} ({existing_count} entries).")
        else:
            with open(cortex_log_path, "w", encoding="utf-8") as f:
                json.dump(cortex_data, f, ensure_ascii=False)
            print(f"Cortex logs synced successfully to {cortex_log_path} ({len(cortex_data)} entries)")
        # Fetch habit descriptions (rule_description) for description-based CLIP comparison
        habits_response = requests.get(f"{server_url}/api/habits_sync")
        habits_response.raise_for_status()
        habit_descriptions = {}
        for h in habits_response.json():
            hname = h.get('name', '').strip()
            full_desc = h.get('description', '')
            parts = full_desc.split(' ||| Prompt: ')
            rule_desc = parts[0].strip() if parts else ''
            if hname and rule_desc:
                habit_descriptions[hname] = rule_desc
        with open("data/habit_descriptions.json", "w", encoding="utf-8") as f:
            json.dump(habit_descriptions, f, ensure_ascii=False, indent=2)
        print(f"Habit descriptions saved to data/habit_descriptions.json ({len(habit_descriptions)} entries)")
    except Exception as e:
        print(f"⚠️  Auto-sync skipped ({type(e).__name__}: {e}). Using existing local data.")
    print("-------------------------\n")

    run_layer_evaluation()
