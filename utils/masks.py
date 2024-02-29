from PIL import Image


class Mask:
    def __init__(self, mask, name=None, identifier=None, ref_image=None):
        self.mask = mask
        self.name = name
        self.identifier = identifier
        self.ref_image = ref_image

    @classmethod
    def from_dict(cls, mask_dict: dict):
        return cls(
            mask=mask_dict["segmentation"]
        )
    
    @classmethod
    def from_list(cls, mask_list: list[dict], ref_image: PIL.Image, names: list[str]=None):
        if names:
            assert len(mask_list) == len(names)

        results = []
        for i in len(mask_list):
            name = names[i] if names else None
            results.append(
                cls(
                    mask=mask_list[i],
                    name=name,
                    identifier=i + 1,
                    ref_image=ref_image
                )
            )
        
        return results
