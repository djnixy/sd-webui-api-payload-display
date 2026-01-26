import os
import json
import time
from typing import Dict, Optional, Any, List
import enum
import traceback
import base64
import io
import shutil  # Added for moving files
from datetime import datetime
import cv2
import gradio as gr
import pydantic
import numpy as np
from PIL import Image

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


def img_to_data_url(img: np.ndarray) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    iobuf = io.BytesIO()
    pil_img.save(iobuf, format="png")
    binary_img = iobuf.getvalue()
    base64_img = base64.b64encode(binary_img)
    base64_img_str = base64_img.decode("utf-8")
    return "data:image/png;base64," + base64_img_str


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

    if isinstance(value, np.ndarray):
        if shared.opts.data.get("api_display_include_base64_images", False):
            return img_to_data_url(value)
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


def api_payload_dict(
    p: StableDiffusionProcessing, api_request: pydantic.BaseModel
) -> Dict:
    excluded_params = [
        "firstphase_width",
        "firstphase_height",
        "sampler_index",
        "send_images",
        "save_images",
    ]

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
            assert isinstance(value, list)
            value = [BASE64_IMAGE_PLACEHOLDER]
        result[name] = value

    return make_json_compatible(result)


def format_payload(payload: Optional[Dict]) -> str:
    if payload is None:
        return "No Payload Found"
    return json.dumps(payload, sort_keys=True, allow_nan=False)


# --- HELPER: Detect tags based on payload ---
def get_payload_tags(payload: Dict) -> str:
    script_name = payload.get("script_name")
    is_xyz = script_name and script_name.lower() == "xyz plot"
    
    alwayson = payload.get("alwayson_scripts", {})
    # Check if any key in alwayson_scripts contains "controlnet"
    is_cnet = any("controlnet" in k.lower() for k in alwayson.keys())

    if is_xyz and is_cnet:
        return "cnet_xyz"
    elif is_xyz:
        return "xyz"
    elif is_cnet:
        return "cnet"
    
    return ""


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
            pull_button = gr.Button(
                visible=False,
                elem_classes=["api-payload-pull"],
                elem_id=f"{process_type_prefix}-api-payload-pull",
            )
            gr.HTML(value='<div class="api-payload-json-tree"></div>')
            self.json_content = gr.Textbox(elem_classes=["api-payload-content"], visible=False)

        pull_button.click(
            lambda: gr.Textbox.update(value=format_payload(self.api_payload)),
            inputs=[],
            outputs=[self.json_content],
        )
        return []

    def process(self, p: StableDiffusionProcessing, *args):
        print(f"[ApiPayloadDisplay] DEBUG: 'process' function STARTED.")
        is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
        api_request = (
            StableDiffusionImg2ImgProcessingAPI
            if is_img2img
            else StableDiffusionTxt2ImgProcessingAPI
        )
        try:
            self.api_payload = api_payload_dict(p, api_request)

            # --- START: SAVE PAYLOAD LOGIC ---
            try:
                script_path = os.path.realpath(__file__)
                base_dir = os.path.dirname(os.path.dirname(script_path))
                payloads_dir = os.path.join(base_dir, "payloads")
                drafts_dir = os.path.join(payloads_dir, "drafts")
                
                os.makedirs(payloads_dir, exist_ok=True)
                
                # Check for High-Res Fix
                enable_hr = self.api_payload.get("enable_hr", False)
                
                if not enable_hr:
                    # Save to drafts
                    os.makedirs(drafts_dir, exist_ok=True)
                    save_dir = drafts_dir
                    tag = ""
                else:
                    # Save to main folder, check for tags
                    save_dir = payloads_dir
                    tag = get_payload_tags(self.api_payload)

                # Construct Filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if tag:
                    filename = f"payload_{timestamp}_{tag}.json"
                else:
                    filename = f"payload_{timestamp}.json"
                
                filepath = os.path.join(save_dir, filename)

                # Save file
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.api_payload, f, indent=4)

                print(f"[ApiPayloadDisplay] DEBUG: Saved payload to: {filepath}")

                # Save "latest" file (Only in root if HR enabled, or maybe just always update latest in root?)
                # To be safe/consistent, let's update payload_latest.json in ROOT only if HR is enabled
                # If you want drafts to update "latest", move this block out of the `if enable_hr` check.
                if enable_hr:
                    latest_filepath = os.path.join(payloads_dir, "payload_latest.json")
                    with open(latest_filepath, "w", encoding="utf-8") as f:
                        json.dump(self.api_payload, f, indent=4)

            except Exception as e:
                print(f"[ApiPayloadDisplay] FATAL Error saving payload: {e}")
            # --- END: SAVE PAYLOAD LOGIC ---

        except Exception as e:
            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            self.api_payload = {"Exception": str(e), "Traceback": "".join(tb_str)}
            print(f"[ApiPayloadDisplay] FATAL Error creating payload: {e}")


def on_ui_settings():
    section = ("api_display", "API Display")
    shared.opts.add_option(
        "api_display_include_base64_images",
        shared.OptionInfo(
            False,
            "Include base64 images in the payload.",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


def organize_existing_payloads():
    """
    Startup logic to reorganize existing files:
    1. If !enable_hr -> Move to drafts.
    2. If enable_hr -> Rename to include _xyz, _cnet, or _cnet_xyz tags if missing.
    """
    script_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    payloads_dir = os.path.join(base_dir, "payloads")
    drafts_dir = os.path.join(payloads_dir, "drafts")

    if not os.path.isdir(payloads_dir):
        return

    os.makedirs(drafts_dir, exist_ok=True)

    for filename in os.listdir(payloads_dir):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(payloads_dir, filename)
        
        # Skip directories
        if not os.path.isfile(filepath):
            continue
            
        # Skip special files
        if filename == "payload_latest.json":
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 1. Check for Drafts (No HR)
            if not data.get("enable_hr", False):
                new_filepath = os.path.join(drafts_dir, filename)
                # Avoid overwrite collision in drafts if possible, or just overwrite
                shutil.move(filepath, new_filepath)
                print(f"[ApiPayloadDisplay] INFO: Moved non-HR payload {filename} to drafts.")
                continue # Done with this file

            # 2. Check for Tags (HR is enabled)
            tag = get_payload_tags(data)
            
            # Check if tag is already in filename
            if tag and tag not in filename:
                name_part, ext_part = os.path.splitext(filename)
                new_filename = f"{name_part}_{tag}{ext_part}"
                new_filepath = os.path.join(payloads_dir, new_filename)
                
                os.rename(filepath, new_filepath)
                print(f"[ApiPayloadDisplay] INFO: Renamed {filename} to {new_filename} (Tag: {tag})")

        except (json.JSONDecodeError, IOError, AttributeError) as e:
            print(f"[ApiPayloadDisplay] ERROR: Could not process {filename}: {e}")


def on_app_started(block, app):
    organize_existing_payloads()


script_callbacks.on_app_started(on_app_started)