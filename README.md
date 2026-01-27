# sd-webui-api-payload-display
Display the corresponding API payload after each generation on WebUI
![Image](https://github.com/huchenlei/sd-webui-api-payload-display/assets/20929282/5fd12cf2-cf94-469f-9525-2bf3622c5237)

# Usage
- Do any generation (txt2img or img2img), after the generation completes, the payload will be populated to the extension accordion.
- Ctrl + click to recursively expand the json tree.
- Click `copy` to copy the json payload to clipboard.

# Features
- Will create dated .json files upon generation.
- if hr_enabled: false, move it to draft path
- will create payload_latest.json, showing the latest json generation
- automatically rename json based on the content, if controlnet was used then the filename will have _cnet_
if xyz plot was used then the filename will have _xyz_
if both, then both pattern will be included.

# Settings
- `api_display_include_base64_images`: Include base64 images in the payload.
- `api_display_startup_deduplicate`: Deduplicate payloads on startup (Deletes older files with identical prompts).

# File Organization
- All generated payloads are stored in the `payloads` directory.
- Payloads generated without `enable_hr` are considered drafts and moved to the `payloads/drafts` directory.
- Payloads are automatically renamed based on their content. If ControlNet is used, the filename will contain `_cnet_`. If XYZ Plot is used, the filename will have `_xyz_`.

# Startup Behavior
- **Organization**: On startup, the script organizes the payloads. It moves drafts to the `drafts` directory and renames files based on their content.
- **Deduplication**: If `api_display_startup_deduplicate` is enabled, the script will delete older payloads that have the same prompt and negative prompt, keeping only the most recent one.

# Special Files
- `payload_latest.json`: A snapshot of the most recent payload generated with `enable_hr`.
- `payload_xyz_skeleton.json`: A skeleton file for XYZ Plot payloads, created when an XYZ Plot is run. The `prompt` is cleared and the `seed` is set to -1.
- `payload_single_skeleton.json`: A skeleton file for single image generation payloads, created when a single image is generated. The `prompt` is cleared and the `seed` is set to -1.

# Deduplication
- **Runtime Deduplication**: To avoid saving the same payload multiple times in quick succession, the script calculates a hash of the payload. If a payload with the same hash was saved less than 2 seconds ago, the new payload is not saved.
- **Startup Deduplication**: As described in the Startup Behavior section, this feature cleans up duplicate payloads based on the prompt.