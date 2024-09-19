from enum import Enum
import numpy as np
import requests
from pydantic import BaseModel
from PIL import Image, ImageShow
import io
import base64


class TaskTypeEnum(str, Enum):
    INPAINTING = "inpainting"
    SEG = "seg"
    DET = "det"
    SCRIBBLE = "scribble"
    AUTOMASK = "automask"
    AUTOMATIC = "automatic"


class InpaintModeEnum(str, Enum):
    MERGE = "merge"
    FIRST = "first"


class ScribbleModeEnum(str, Enum):
    MERGE = "merge"
    SPLIT = "split"


class GroundedSAMOutputMsg(BaseModel):
    full_image: str
    mask_image: str | None = None
    masks: list[str] | None = None


class GroundedSAMOutput:
    full_image: Image.Image
    mask_image: Image.Image | None
    masks: list[np.ndarray] | None

    def __init__(
        self,
        full_image: Image.Image,
        mask_image: Image.Image | None,
        masks: list[np.ndarray] | None,
    ):
        self.full_image = full_image
        self.mask_image = mask_image
        self.masks = masks

    @staticmethod
    def img_to_base64_str(img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def base64_str_to_img(base64_str: str) -> Image.Image:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))

    @staticmethod
    def numpy_to_base64_str(array: np.ndarray) -> str:
        # expected input to be rgb image, not bgr
        buffered = io.BytesIO()
        img = Image.fromarray(array)
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def base64_str_to_numpy(base64_str: str) -> np.ndarray:
        array_data = base64.b64decode(base64_str)
        buffered = io.BytesIO(array_data)
        return np.load(buffered)

    def to_model(self):
        return GroundedSAMOutputMsg(
            full_image=self.img_to_base64_str(self.full_image),
            mask_image=(
                self.img_to_base64_str(self.mask_image) if self.mask_image else None
            ),
            masks=(
                [self.numpy_to_base64_str(mask) for mask in self.masks]
                if self.masks
                else None
            ),
        )

    def to_dict(self):
        return self.to_model().model_dump()

    @classmethod
    def from_dict(cls, data):
        full_image = cls.base64_str_to_img(data["full_image"])
        mask_image = (
            cls.base64_str_to_img(data["mask_image"]) if data["mask_image"] else None
        )
        masks = (
            [cls.base64_str_to_numpy(mask) for mask in data["masks"]]
            if data["masks"]
            else None
        )
        return cls(full_image, mask_image, masks)


class GroundedSAMRestful:
    def __init__(self, endpoint: str = "http://127.0.0.1:8080"):
        self.endpoint = endpoint

    def call_with_numpy(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        text_prompt: str = "",
        task_type: TaskTypeEnum | str = TaskTypeEnum.SEG,
        inpaint_prompt: str | None = None,
        box_threshold: float | None = 0.3,
        text_threshold: float | None = 0.25,
        iou_threshold: float | None = 0.5,
        inpaint_mode: InpaintModeEnum | str | None = None,
        scribble_mode: ScribbleModeEnum | str | None = None,
        openai_api_key: str | None = None,
    ):

        image_bytes = GroundedSAMOutput.numpy_to_base64_str(image)
        mask_bytes = (
            GroundedSAMOutput.numpy_to_base64_str(mask) if mask is not None else None
        )

        return self._make_api_call(
            image_bytes,
            mask_bytes,
            text_prompt,
            task_type,
            inpaint_prompt,
            box_threshold,
            text_threshold,
            iou_threshold,
            inpaint_mode,
            scribble_mode,
            openai_api_key,
        )

    def call_with_filepath(
        self,
        image: str,
        mask: str | None = None,
        text_prompt: str = "",
        task_type: TaskTypeEnum | str = TaskTypeEnum.SEG,
        inpaint_prompt: str | None = None,
        box_threshold: float | None = 0.3,
        text_threshold: float | None = 0.25,
        iou_threshold: float | None = 0.5,
        inpaint_mode: InpaintModeEnum | str | None = None,
        scribble_mode: ScribbleModeEnum | str | None = None,
        openai_api_key: str | None = None,
    ):
        image = Image.open(image)
        mask = Image.open(mask) if mask is not None else None

        image_bytes = GroundedSAMOutput.img_to_base64_str(image)
        mask_bytes = (
            GroundedSAMOutput.img_to_base64_str(mask) if mask is not None else None
        )

        return self._make_api_call(
            image_bytes,
            mask_bytes,
            text_prompt,
            task_type,
            inpaint_prompt,
            box_threshold,
            text_threshold,
            iou_threshold,
            inpaint_mode,
            scribble_mode,
            openai_api_key,
        )

    def _make_api_call(
        self,
        image_bytes: str,
        mask_bytes: str | None = None,
        text_prompt: str = "",
        task_type: TaskTypeEnum | str = TaskTypeEnum.SEG,
        inpaint_prompt: str | None = None,
        box_threshold: float | None = 0.3,
        text_threshold: float | None = 0.25,
        iou_threshold: float | None = 0.5,
        inpaint_mode: InpaintModeEnum | str | None = None,
        scribble_mode: ScribbleModeEnum | str | None = None,
        openai_api_key: str | None = None,
    ):
        _path = "/v1/grounded_sam"

        payload = {
            "input_image": {"image": image_bytes, "mask": mask_bytes},
            "text_prompt": text_prompt,
            "task_type": (
                task_type.value if isinstance(task_type, TaskTypeEnum) else task_type
            ),  # Example task type; use as per your requirement
            "inpaint_prompt": inpaint_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "iou_threshold": iou_threshold,
            "inpaint_mode": (
                inpaint_mode.value
                if isinstance(inpaint_mode, InpaintModeEnum)
                else inpaint_mode
            ),
            "scribble_mode": (
                scribble_mode.value
                if isinstance(inpaint_mode, InpaintModeEnum)
                else scribble_mode
            ),
            "openai_api_key": openai_api_key,
        }

        response = requests.post(f"{self.endpoint}{_path}", json=payload, timeout=60)
        return GroundedSAMOutput.from_dict(response.json())

if __name__ == "__main__":
    api = GroundedSAMRestful("http://127.0.0.1:7584")
    res = api.call_with_filepath(
        "test.jpg", text_prompt="blue tape", task_type=TaskTypeEnum.SEG
    )
    print(res.masks)
    ImageShow.show(res.full_image)
