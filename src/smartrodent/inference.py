import speciesnet
import yolo26
import torch


# This would work with faunanet abstractions.
class Preprocessor:
    def __init__(
        self,
    ): ...

    def process(self, imgs): ...


class InferencePipeline:
    def __init__(
        self,
        detector_model: str,
        classifier_model: str,
        classes: list[str],
        class_mappings: dict[str, str],
    ):
        self.classes = classes
        self.class_mappings = class_mappings

    def _merge_split_classes(self, classification_results): ...

    def _detect(self, img):
        ...
        # run detector, extract crops

    def _classify(self, crop):
        ...
        # take single crop, pass to classifier

    def _process_event(self, imgs):
        ...
        # do detect->classify cycle for a single image

    def _process_sequence(self, imgs):
        return [self._process_event(img) for img in imgs]

    def _estimate_uncertainty(self, imgs):
        ...
        # estimate how certain we are for the entire sequence to
        # - belong to a single event
        # - belong to a single class

    def process(self, imgs):
        results = self._process_sequence(imgs)
        uncertainty = self._estimate_uncertainty(results)

        return results, uncertainty
