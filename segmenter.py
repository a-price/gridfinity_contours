import torch
from numpy.typing import NDArray
from transformers import Sam2Model, Sam2Processor


class Segmenter:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny").to(
            self.device  # pyright: ignore[reportArgumentType]
        )
        self.processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-tiny")

    # Returns a set of mask hypotheses with tensor indices:
    # [image, object, point, coordinates=2]
    def Segment(
        self,
        image,
        input_points: list[list[list[list[float]]]] | None,
        input_labels: list[list[list[int]]] | None,
    ) -> NDArray:
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        return masks.detach().cpu().numpy()
