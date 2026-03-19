# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""SimLingo VLA wrapper implementing the common AlpaSim driver interface.

SimLingo (CVPR'25) is a Vision-Language-Action model trained on CARLA data
that uses InternVL2-1B as the vision backbone with a LoRA-adapted LLM decoder
predicting 11 waypoints at 5 Hz.

Reference: https://github.com/RenzKa/simlingo
Model:     https://huggingface.co/RenzKa/simlingo
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoProcessor

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction

logger = logging.getLogger(__name__)

# Default simlingo repo path — can be overridden via SIMLINGO_REPO_PATH env var
_DEFAULT_SIMLINGO_REPO_PATH = os.environ.get(
    "SIMLINGO_REPO_PATH", "/workspace/simlingo"
)

# Ego-relative command text used in the prompt (CARLA convention)
_COMMAND_TEXT = {
    DriveCommand.LEFT: "go left at the next intersection",
    DriveCommand.RIGHT: "go right at the next intersection",
    DriveCommand.STRAIGHT: "follow the road",
    DriveCommand.UNKNOWN: "follow the road",
}

# ImageNet normalisation constants (InternVL2 preprocessing)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing helpers (re-implemented locally to avoid CARLA imports)
# ---------------------------------------------------------------------------


def _build_transform(input_size: int = 448) -> T.Compose:
    """Return the InternVL2 image transform (resize + normalise)."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios, width: int, height: int, image_size: int
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 2,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Tile a PIL image into InternVL2 patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_w, target_h))
    tiles: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


# ---------------------------------------------------------------------------
# SimLingo camera helpers (extracted from simlingo/team_code/simlingo_utils.py)
# ---------------------------------------------------------------------------


def _get_camera_intrinsics(w: int, h: int, fov: float = 110.0) -> torch.Tensor:
    """Return a [3, 3] float32 intrinsics matrix for the CARLA camera."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return torch.tensor(K)


def _get_camera_extrinsics() -> torch.Tensor:
    """Return a [4, 4] float32 extrinsics matrix for the CARLA front camera."""
    ext = np.zeros((4, 4), dtype=np.float32)
    ext[3, 3] = 1.0
    ext[:3, :3] = np.eye(3)
    ext[:3, 3] = [-1.5, 0.0, 2.0]
    return torch.tensor(ext)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class SimLingoModel(BaseTrajectoryModel):
    """SimLingo wrapper implementing the common AlpaSim driver interface.

    SimLingo is a Vision-Language-Action model (CVPR'25) that uses InternVL2-1B
    as its vision backbone and predicts 11 waypoints at 5 Hz from a single
    front-facing camera image.

    The model was trained on CARLA synthetic data; a domain gap exists when
    deploying against NuRec real-world reconstructions.

    Coordinate convention:
        Waypoints are output in the CARLA ego frame (X = forward, Y = left).
        This is consistent with AlpaSim's rig frame convention.
    """

    # InternVL2 tile size
    IMAGE_SIZE = 448
    # Maximum tiles per image (2 = 1 tile + 1 global thumbnail when USE_GLOBAL_IMG)
    MAX_NUM_TILES = 2
    # Waypoints predicted per step (training pred_len = 11)
    NUM_WAYPOINTS = 11
    # Output frequency matches CARLA 5 Hz data collection (waypoint every 0.2 s)
    OUTPUT_FREQUENCY_HZ = 5
    # SimLingo is a single-frame model (hist_len = 1 in training config)
    CONTEXT_LENGTH = 1
    # Crop bottom 4.8/16 of the image (matches training preprocessing)
    CUT_BOTTOM_FRACTION = 4.8 / 16.0

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        simlingo_repo_path: str = _DEFAULT_SIMLINGO_REPO_PATH,
    ):
        """Initialise SimLingo.

        Args:
            checkpoint_path: Absolute path to *pytorch_model.pt* state-dict file.
                The Hydra training config is expected at
                ``<ckpt_dir>/../../.hydra/config.yaml``.
            device: Torch device for inference.
            camera_ids: List containing exactly **one** camera ID (front camera).
            simlingo_repo_path: Root of the simlingo git repo (needed for imports).
        """
        if len(camera_ids) != 1:
            raise ValueError(
                f"SimLingo requires exactly 1 camera, got {len(camera_ids)}: {camera_ids}"
            )

        # Make simlingo importable
        if simlingo_repo_path not in sys.path:
            sys.path.insert(0, simlingo_repo_path)
            logger.info("Added %s to sys.path for simlingo imports", simlingo_repo_path)

        self._device = device
        self._camera_ids = camera_ids
        self._simlingo_repo_path = simlingo_repo_path

        # ------------------------------------------------------------------ #
        # Locate the Hydra config
        # ------------------------------------------------------------------ #
        # The checkpoint lives at:
        #   <outputs_root>/<run>/checkpoints/<epoch_dir>/pytorch_model.pt
        # so .hydra/config.yaml is 3 levels up at <outputs_root>/<run>/.hydra/config.yaml
        ckpt_path = Path(checkpoint_path)
        candidates = [
            ckpt_path.parent.parent.parent / ".hydra" / "config.yaml",  # 3 levels up
            ckpt_path.parent.parent / ".hydra" / "config.yaml",          # 2 levels up
            ckpt_path.parent / ".hydra" / "config.yaml",                 # 1 level up
        ]
        hydra_cfg_path = next((c for c in candidates if c.exists()), None)

        if hydra_cfg_path is None:
            # Last resort: search the entire outputs directory
            outputs_dir = Path(simlingo_repo_path) / "outputs"
            if outputs_dir.exists():
                for p in sorted(outputs_dir.rglob(".hydra")):
                    cand = p / "config.yaml"
                    if cand.exists():
                        hydra_cfg_path = cand
                        break

        if hydra_cfg_path is None:
            raise FileNotFoundError(
                f"Could not find Hydra config for checkpoint at {checkpoint_path}. "
                "Searched 1-3 levels above the checkpoint for a '.hydra/config.yaml'. "
                f"Tried: {[str(c) for c in candidates]}"
            )

        logger.info("Loading Hydra config from %s", hydra_cfg_path)
        cfg = OmegaConf.load(hydra_cfg_path)

        # use_global_img controls whether a thumbnail tile is appended
        self._use_global_img = bool(
            cfg.get("data_module", {}).get("use_global_img", False)
        )

        # ------------------------------------------------------------------ #
        # Processor (tokenizer) and special tokens
        # ------------------------------------------------------------------ #
        vision_variant: str = cfg.model.vision_model.variant
        cache_dir = str(
            Path(simlingo_repo_path) / "pretrained" / vision_variant.split("/")[1]
        )
        pretrained_source = cache_dir if Path(cache_dir).exists() else vision_variant
        logger.info("Loading processor from %s", pretrained_source)

        processor = AutoProcessor.from_pretrained(
            pretrained_source, trust_remote_code=True
        )
        self._tokenizer = (
            processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        )
        self._tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<WAYPOINTS>",
                    "<WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS>",
                    "<WAYPOINT_LAST>",
                    "<ROUTE>",
                    "<ROUTE_DIFF>",
                    "<TARGET_POINT>",
                ]
            }
        )
        self._tokenizer.padding_side = "left"

        # Number of visual tokens per image (used to build the prompt template)
        tmp_cfg = AutoConfig.from_pretrained(pretrained_source, trust_remote_code=True)
        img_sz = tmp_cfg.force_image_size or tmp_cfg.vision_config.image_size
        patch_sz = tmp_cfg.vision_config.patch_size
        self._num_image_token = int(
            (img_sz // patch_sz) ** 2 * (tmp_cfg.downsample_ratio ** 2)
        )

        # ------------------------------------------------------------------ #
        # Conversation template (used by InternVL2 / InternLM2)
        # ------------------------------------------------------------------ #
        self._conv_module = self._load_conv_module(cache_dir, vision_variant)

        # ------------------------------------------------------------------ #
        # Instantiate the DrivingModel via Hydra
        # ------------------------------------------------------------------ #
        import hydra  # noqa: PLC0415

        # Override the vision/language model variant to the absolute local path.
        # This prevents AutoModel.from_pretrained() from trying to reach HuggingFace
        # when the weights are already cached under pretrained/<model_name>/.
        if Path(cache_dir).exists():
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            cfg_dict["model"]["vision_model"]["variant"] = cache_dir
            cfg_dict["model"]["language_model"]["variant"] = cache_dir
            cfg = OmegaConf.create(cfg_dict)
            logger.info("Overriding model variant to local path: %s", cache_dir)

        logger.info("Instantiating DrivingModel for %s", vision_variant)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self._model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            _recursive_=False,
        ).to(device)
        torch.set_default_dtype(default_dtype)

        # ------------------------------------------------------------------ #
        # Load fine-tuned state dict
        # ------------------------------------------------------------------ #
        logger.info("Loading state dict from %s", checkpoint_path)
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self._model.load_state_dict(state_dict)
        self._model.eval()

        # Image preprocessing transform
        self._transform = _build_transform(self.IMAGE_SIZE)

        logger.info(
            "SimLingoModel ready — camera=%s device=%s", camera_ids[0], device
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_conv_module(self, cache_dir: str, variant: str):
        """Import the InternVL2 conversation-template module from the cache dir."""
        model_path = Path(cache_dir) / "conversation.py"
        if not model_path.exists():
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            logger.info("Downloading model files from %s …", variant)
            snapshot_download(repo_id=variant, local_dir=cache_dir)
        spec = importlib.util.spec_from_file_location(
            "get_conv_template", str(model_path)
        )
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = conv_module
        spec.loader.exec_module(conv_module)
        return conv_module

    def _preprocess_image(self, image_hwc: np.ndarray) -> torch.Tensor:
        """Preprocess a single HWC uint8 RGB image to InternVL2 tile format.

        Applies:
        1. JPEG round-trip (matches training-time JPEG compression artifacts).
        2. Bottom-quarter crop.
        3. InternVL2 dynamic tiling → normalised float tiles.

        Returns:
            Tensor of shape ``[1, num_patches, C, H, W]``.
        """
        # JPEG round-trip
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Bottom crop
        h = img.shape[0]
        cut = int(h * self.CUT_BOTTOM_FRACTION)
        img = img[: h - cut, :, :]

        pil = Image.fromarray(img)
        tiles = _dynamic_preprocess(
            pil,
            image_size=self.IMAGE_SIZE,
            use_thumbnail=self._use_global_img,
            max_num=self.MAX_NUM_TILES,
        )
        pixel_values = torch.stack([self._transform(t) for t in tiles])  # [P, C, H, W]
        return pixel_values.unsqueeze(0)  # [1, P, C, H, W]

    def _build_prompt_label(self, speed: float, command: DriveCommand):
        """Build a tokenised ``LanguageLabel`` for the model.

        Uses command-only prompting (no target-point token), which is supported
        by all SimLingo variants.
        """
        from simlingo_training.utils.custom_types import LanguageLabel  # noqa: PLC0415

        cmd_text = _COMMAND_TEXT.get(command, "follow the road")
        prompt = (
            f"Current speed: {speed:.1f} m/s. "
            f"Command: {cmd_text}. "
            "What should the ego do next?"
        )

        # Build the InternLM2 chat template (inference: assistant turn = None)
        template = self._conv_module.get_conv_template("internlm2-chat")
        user_content = "<image>\n" + prompt
        template.append_message(template.roles[0], user_content)
        template.append_message(template.roles[1], None)

        query = template.get_prompt()
        # Strip system prompt to save tokens (matches agent_simlingo.py)
        system_prompt = (
            template.system_template.replace("{system_message}", template.system_message)
            + template.sep
        )
        query = query.replace(system_prompt, "")

        # Replace <image> with InternVL2 image-context tokens
        num_patches = self.MAX_NUM_TILES
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self._num_image_token * num_patches
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)

        tokenized = self._tokenizer(
            [query],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = tokenized["input_ids"]
        valid = ids != self._tokenizer.pad_token_id

        return LanguageLabel(
            phrase_ids=ids.to(self._device),
            phrase_valid=valid.to(self._device),
            phrase_mask=valid.to(self._device),
            placeholder_values=[{}],
            language_string=[query],
            loss_masking=None,
        )

    # ------------------------------------------------------------------ #
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------ #

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self.CONTEXT_LENGTH

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> str:
        return _COMMAND_TEXT.get(command, "follow the road")

    def predict(
        self,
        camera_images: dict[str, list[tuple[int, np.ndarray]]],
        command: DriveCommand,
        speed: float,
        acceleration: float,
        ego_pose_at_time_history_local: list[Any] | None = None,
    ) -> ModelPrediction:
        """Predict a trajectory from the latest camera frame.

        Args:
            camera_images: ``{camera_id: [(timestamp_us, image_hwc), ...]}``.
                           The list must have length == ``context_length`` (1).
            command: Navigation command from the route planner.
            speed: Current vehicle speed in m/s.
            acceleration: Unused by SimLingo.
            ego_pose_at_time_history_local: Unused by SimLingo.

        Returns:
            ``ModelPrediction`` with ``trajectory_xy`` of shape ``(11, 2)`` in
            rig frame (X = forward, Y = left).
        """
        self._validate_cameras(camera_images)

        cam_id = self._camera_ids[0]
        frames = camera_images[cam_id]
        if not frames:
            logger.warning("SimLingo: no frames received — returning zero trajectory")
            return ModelPrediction(
                trajectory_xy=np.zeros((self.NUM_WAYPOINTS, 2)),
                headings=np.zeros(self.NUM_WAYPOINTS),
            )

        _, image_hwc = frames[-1]  # most recent frame

        # ---- Image preprocessing ---------------------------------------- #
        pixel_values = self._preprocess_image(image_hwc)          # [1, P, C, H, W]
        _B, num_patches, C, H, W = pixel_values.shape
        # Reshape to [B=1, T=1, num_patches, C, H, W] as model expects
        processed_image = pixel_values.view(1, 1, num_patches, C, H, W)

        # ---- Prompt ----------------------------------------------------- #
        ll = self._build_prompt_label(speed, command)

        # ---- DrivingInput ----------------------------------------------- #
        from simlingo_training.utils.custom_types import DrivingInput  # noqa: PLC0415

        driving_input = DrivingInput(
            camera_images=processed_image.to(self._device).bfloat16(),
            image_sizes=torch.tensor([[H, W]]),
            camera_intrinsics=_get_camera_intrinsics(W, H).unsqueeze(0).view(1, 3, 3).float().to(self._device),
            camera_extrinsics=_get_camera_extrinsics().unsqueeze(0).view(1, 4, 4).float().to(self._device),
            vehicle_speed=torch.tensor([[speed]], device=self._device, dtype=torch.float32),
            target_point=torch.zeros(1, 2, device=self._device, dtype=torch.float32),
            prompt=ll,
            prompt_inference=ll,
        )

        # ---- Inference -------------------------------------------------- #
        with torch.no_grad():
            pred_speed_wps, _pred_route, language = self._model(driving_input)

        if pred_speed_wps is None:
            logger.warning("SimLingo returned None waypoints — returning zeros")
            return ModelPrediction(
                trajectory_xy=np.zeros((self.NUM_WAYPOINTS, 2)),
                headings=np.zeros(self.NUM_WAYPOINTS),
            )

        # pred_speed_wps: [1, NUM_WAYPOINTS, 2]
        trajectory_xy: np.ndarray = (
            pred_speed_wps[0].float().detach().cpu().numpy()
        )
        headings = self._compute_headings_from_trajectory(trajectory_xy)

        reasoning_text: str | None = language[0] if language else None
        if reasoning_text:
            logger.debug("SimLingo response: %s", reasoning_text)

        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )
