import os
import json
import time
from typing import Dict, Optional, Any, List
import enum
import traceback
import base64
import io
import shutil
from datetime import datetime
import cv2
import gradio as gr
import pydantic
import numpy as np
from PIL import Image
import hashlib
import re

import modules.scripts as scripts
from modules import shared, script_callbacks
from modules.api.models import (
    StableDiffusionImg2ImgProcessingAPI,
    StableDiffusionTxt2ImgProcessingAPI,
)
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)

BASE64_IMAGE_PLACEHOLDER = "base64image placeholder"

# --- GLOBAL VARS FOR RUNTIME DEDUPLICATION ---
_LAST_PAYLOAD_HASH = None
_LAST_SAVE_TIME = 0

# --- HELPERS ---
def pil_to_data_url(pil_img: Image.Image) -> str:
    iobuf = io.BytesIO()
    pil_img.save(iobuf, format="png")
    binary_img = iobuf.getvalue()
    base64_img = base64.b64encode(binary_img)
    base64_img_str = base64_img.decode("utf-8")
    return "data:image/png;base64," + base64_img_str

def img_to_data_url(img: np.ndarray) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil_to_data_url(pil_img)

def make_json_compatible(value: Any) -> Any:
    def is_jsonable(x):
        try:
            json.dumps(x, allow_nan=False)
            return True
        except (TypeError, OverflowError, ValueError):
            return False

    if is_jsonable(value):
        return value

    if isinstance(value, dict):
        return {k: make_json_compatible(v) for k, v in value.items()}

    if any(isinstance(value, t) for t in (set, list, tuple)):
        return [make_json_compatible(v) for v in value]

    if isinstance(value, enum.Enum):
        return make_json_compatible(value.value)

    include_images = shared.opts.data.get("api_display_include_base64_images", True)

    if isinstance(value, np.ndarray):
        if include_images:
            return img_to_data_url(value)
        else:
            return BASE64_IMAGE_PLACEHOLDER

    if isinstance(value, Image.Image):
        if include_images:
            return pil_to_data_url(value)
        else:
            return BASE64_IMAGE_PLACEHOLDER

    if hasattr(value, "__dict__"):
        return make_json_compatible(vars(value))

    if value in (float("inf"), float("-inf")):
        return None

    return None

def selectable_script_payload(p: StableDiffusionProcessing) -> Dict:
    script_runner: scripts.ScriptRunner = p.scripts
    selectable_script_index = p.script_args[0]
    
    if selectable_script_index == 0:
        return {"script_name": None, "script_args": []}

    selectable_script: scripts.Script = script_runner.selectable_scripts[
        selectable_script_index - 1
    ]
    title = selectable_script.title()
    
    return {
        "script_name": title.lower()
        if title
        else os.path.basename(selectable_script.filename).lower(),
        "script_args": p.script_args[
            selectable_script.args_from : selectable_script.args_to
        ],
    }

def alwayson_script_payload(p: StableDiffusionProcessing) -> Dict:
    script_runner: scripts.ScriptRunner = p.scripts
    all_scripts: Dict[str, List] = {}
    for alwayson_script in script_runner.alwayson_scripts:
        title = alwayson_script.title()
        all_scripts[
            title.lower() if title else os.path.basename(alwayson_script.filename).lower()
        ] = {"args": p.script_args[alwayson_script.args_from : alwayson_script.args_to]}
    return {"alwayson_scripts": all_scripts}

def seed_enable_extras_payload(p: StableDiffusionProcessing) -> Dict:
    return {
        "seed_enable_extras": not (
            p.subseed == -1
            and p.subseed_strength == 0
            and p.seed_resize_from_h == 0
            and p.seed_resize_from_w == 0
        )
    }

def api_payload_dict(p: StableDiffusionProcessing, api_request: pydantic.BaseModel) -> Dict:
    excluded_params = ["firstphase_width", "firstphase_height", "sampler_index", "send_images", "save_images"]
    result = {}
    result.update(selectable_script_payload(p))
    result.update(alwayson_script_payload(p))
    result.update(seed_enable_extras_payload(p))

    for name in api_request.__fields__.keys():
        if name in result or name in excluded_params:
            continue
        if not hasattr(p, name):
            continue
        value = getattr(p, name)
        if value is None:
            continue
        
        if isinstance(p, StableDiffusionProcessingImg2Img) and name == "init_images":
            if isinstance(value, list):
                value = [make_json_compatible(v) for v in value]
            else:
                 value = [BASE64_IMAGE_PLACEHOLDER]
        
        result[name] = value
    return make_json_compatible(result)

def format_payload(payload: Optional[Dict]) -> str:
    if payload is None:
        return "No Payload Found"
    return json.dumps(payload, sort_keys=True, allow_nan=False)

def get_payload_tags(payload: Dict) -> str:
    script_name = payload.get("script_name")
    is_xyz = script_name and script_name.lower() == "x/y/z plot"
    
    is_cnet = False
    alwayson = payload.get("alwayson_scripts", {})
    
    for key, data in alwayson.items():
        if "controlnet" in key.lower():
            args = data.get("args", [])
            for unit in args:
                if isinstance(unit, dict) and unit.get("enabled", False) is True:
                    is_cnet = True
                    break
        if is_cnet:
            break

    if is_xyz and is_cnet:
        return "cnet_xyz"
    elif is_xyz:
        return "xyz"
    elif is_cnet:
        return "cnet"
    return ""

# --- MAIN SCRIPT CLASS ---
class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.json_content: Optional[gr.HTML] = None
        self.api_payload: Optional[Dict] = None

    def title(self) -> str:
        return "API payload"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> List:
        process_type_prefix = "img2img" if is_img2img else "txt2img"
        with gr.Accordion(f"API payload", open=False, elem_classes=["api-payload-display"]):
            pull_button = gr.Button(visible=False, elem_classes=["api-payload-pull"], elem_id=f"{process_type_prefix}-api-payload-pull")
            gr.HTML(value='<div class="api-payload-json-tree"></div>')
            self.json_content = gr.Textbox(elem_classes=["api-payload-content"], visible=False)

        pull_button.click(
            lambda: gr.Textbox.update(value=format_payload(self.api_payload)),
            inputs=[],
            outputs=[self.json_content],
        )
        return []

    def process(self, p: StableDiffusionProcessing, *args):
        global _LAST_PAYLOAD_HASH, _LAST_SAVE_TIME
        
        print(f"[ApiPayloadDisplay] DEBUG: 'process' function STARTED.")
        is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
        api_request = StableDiffusionImg2ImgProcessingAPI if is_img2img else StableDiffusionTxt2ImgProcessingAPI
        
        # Determine Mode for Filename
        mode_tag = "img2img" if is_img2img else "txt2img"

        try:
            self.api_payload = api_payload_dict(p, api_request)
            
            # Runtime Deduplication
            payload_str = json.dumps(self.api_payload, sort_keys=True)
            current_hash = hashlib.md5(payload_str.encode('utf-8')).hexdigest()
            current_time = time.time()

            if _LAST_PAYLOAD_HASH == current_hash and (current_time - _LAST_SAVE_TIME) < 2.0:
                print("[ApiPayloadDisplay] INFO: Skipping duplicate payload save.")
                return
            
            _LAST_PAYLOAD_HASH = current_hash
            _LAST_SAVE_TIME = current_time

            # Save Logic
            try:
                script_path = os.path.realpath(__file__)
                base_dir = os.path.dirname(os.path.dirname(script_path))
                payloads_dir = os.path.join(base_dir, "payloads")
                drafts_dir = os.path.join(payloads_dir, "drafts")
                
                os.makedirs(payloads_dir, exist_ok=True)
                
                enable_hr = self.api_payload.get("enable_hr", False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if not enable_hr:
                    os.makedirs(drafts_dir, exist_ok=True)
                    save_dir = drafts_dir
                    filename = f"payload_{mode_tag}_{timestamp}.json"
                else:
                    save_dir = payloads_dir
                    tag = get_payload_tags(self.api_payload)
                    if tag:
                        filename = f"payload_{mode_tag}_{tag}_{timestamp}.json"
                    else:
                        filename = f"payload_{mode_tag}_{timestamp}.json"

                filepath = os.path.join(save_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.api_payload, f, indent=4)

                print(f"[ApiPayloadDisplay] DEBUG: Saved payload to: {filepath}")

                # --- SKELETON & LATEST LOGIC (Only if HR Enabled) ---
                if enable_hr:
                    # 1. Update payload_latest.json
                    latest_filepath = os.path.join(payloads_dir, "payload_latest.json")
                    with open(latest_filepath, "w", encoding="utf-8") as f:
                        json.dump(self.api_payload, f, indent=4)

                    # 2. Create Skeleton File
                    script_name = self.api_payload.get("script_name", "")
                    is_xyz = script_name and script_name.lower() == "x/y/z plot"

                    if is_xyz:
                        skeleton_filename = "payload_xyz_skeleton.json"
                    else:
                        skeleton_filename = "payload_single_skeleton.json"

                    # Create a copy to modify
                    skeleton_payload = self.api_payload.copy()
                    
                    # Apply Skeleton Rules: Remove prompt, Seed -1
                    skeleton_payload["prompt"] = ""
                    skeleton_payload["seed"] = -1
                    
                    skeleton_filepath = os.path.join(payloads_dir, skeleton_filename)
                    with open(skeleton_filepath, "w", encoding="utf-8") as f:
                        json.dump(skeleton_payload, f, indent=4)
                    
                    print(f"[ApiPayloadDisplay] DEBUG: Updated skeleton file: {skeleton_filename}")

            except Exception as e:
                print(f"[ApiPayloadDisplay] FATAL Error saving payload: {e}")

        except Exception as e:
            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            self.api_payload = {"Exception": str(e), "Traceback": "".join(tb_str)}
            print(f"[ApiPayloadDisplay] FATAL Error creating payload: {e}")

# --- SETTINGS ---
def on_ui_settings():
    section = ("api_display", "API Display")
    shared.opts.add_option(
        "api_display_include_base64_images",
        shared.OptionInfo(True, "Include base64 images in the payload.", gr.Checkbox, {"interactive": True}, section=section),
    )
    shared.opts.add_option(
        "api_display_startup_deduplicate",
        shared.OptionInfo(False, "Deduplicate payloads on startup (Deletes older files with identical prompts).", gr.Checkbox, {"interactive": True}, section=section),
    )

script_callbacks.on_ui_settings(on_ui_settings)

# --- STARTUP LOGIC ---
def get_file_timestamp_key(filename):
    match = re.search(r"(\d{8}_\d{6})", filename)
    if match:
        return match.group(1)
    return "0"

def organize_and_deduplicate():
    script_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    payloads_dir = os.path.join(base_dir, "payloads")
    drafts_dir = os.path.join(payloads_dir, "drafts")

    if not os.path.isdir(payloads_dir):
        return
    
    os.makedirs(drafts_dir, exist_ok=True)
    
    do_deduplicate = shared.opts.data.get("api_display_startup_deduplicate", False)
    
    # --- PROTECTED FILES LIST (Crucial for ensuring skeletons are ignored) ---
    PROTECTED_FILES = ["payload_latest.json", "payload_single_skeleton.json", "payload_xyz_skeleton.json"]

    # 1. ORGANIZE
    print("[ApiPayloadDisplay] Starting Organization...")
    for filename in os.listdir(payloads_dir):
        if not filename.endswith(".json"): continue
        if filename in PROTECTED_FILES: continue  # SKIP SKELETONS & LATEST

        # Optimization: Skip if already tagged correctly with MODE (txt2img/img2img)
        # This prevents checking every single file every boot once they are migrated.
        if "_txt2img_" in filename or "_img2img_" in filename:
             continue 
        
        filepath = os.path.join(payloads_dir, filename)
        if not os.path.isfile(filepath): continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Move Drafts (Non-HR)
            if not data.get("enable_hr", False):
                new_filepath = os.path.join(drafts_dir, filename)
                shutil.move(filepath, new_filepath)
                continue 
            
            # Detect Mode from JSON content
            # "init_images" is only present in Img2Img payloads
            mode_tag = "img2img" if "init_images" in data else "txt2img"

            tag = get_payload_tags(data)
            match = re.search(r"(\d{8}_\d{6})", filename)
            if match:
                timestamp = match.group(1)
                
                # Construct New Filename with Mode
                if tag:
                    new_filename = f"payload_{mode_tag}_{tag}_{timestamp}.json"
                else:
                    new_filename = f"payload_{mode_tag}_{timestamp}.json"
                
                if filename != new_filename:
                    new_filepath = os.path.join(payloads_dir, new_filename)
                    os.rename(filepath, new_filepath)

        except (json.JSONDecodeError, IOError, AttributeError):
            continue

    # 2. DEDUPLICATE
    if do_deduplicate:
        print("[ApiPayloadDisplay] Starting Deduplication...")
        prompt_groups = {}
        
        files = [f for f in os.listdir(payloads_dir) if f.endswith(".json") and f not in PROTECTED_FILES]
        
        for filename in files:
            filepath = os.path.join(payloads_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                pos = data.get("prompt", "").strip()
                neg = data.get("negative_prompt", "").strip()
                prompts = (pos, neg)
                
                if prompts not in prompt_groups:
                    prompt_groups[prompts] = []
                prompt_groups[prompts].append(filename)
            except:
                continue
        
        del_count = 0
        for prompts, file_list in prompt_groups.items():
            if len(file_list) < 2: continue
            
            file_list.sort(key=get_file_timestamp_key)
            
            to_delete = file_list[:-1]
            for file_to_remove in to_delete:
                try:
                    os.remove(os.path.join(payloads_dir, file_to_remove))
                    del_count += 1
                except: pass
        
        if del_count > 0:
            print(f"[ApiPayloadDisplay] Cleanup: Deleted {del_count} duplicate payloads.")

def on_app_started(block, app):
    organize_and_deduplicate()

script_callbacks.on_app_started(on_app_started)