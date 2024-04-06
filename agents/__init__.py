from .agent import Agent

from .seg_vlm import SegVLM
from .det_vlm import DetVLM
from .det_llm import DetLLM
from .vlm_det import VLMDet
from .vlm_det_inspect import VLMDetInspect
from .dom import DOM


def agent_factory(agent_type, segmentor=None, vlm=None, detector=None, llm=None, configs=None, logger=None):
    """
    Factory method to create an instance of a specific Agent subclass with default values.

    Args:
        agent_type (str): The type of agent to create. Possible values are 'SegVLM', 'DetVLM', and 'DetLLM'.
        segmentor (Segmentor, optional): An instance of Segmentor. Defaults to a default instance if not provided.
        vlm (LanguageModel, optional): An instance of LanguageModel for VLM. Defaults to a default instance if not provided.
        detector (Detector, optional): An instance of Detector. Defaults to a default instance if not provided.
        llm (LanguageModel, optional): An instance of LanguageModel for LLM. Defaults to a default instance if not provided.
        configs (dict, optional): A dictionary of configuration settings.

    Returns:
        Agent: An instance of the specified Agent subclass.
    """

    # Use default instances if none are provided
    from apis.detectors import OWLViT
    from apis.segmentors import SAM
    from apis.language_model import GPT4, GPT4V
    segmentor = segmentor or SAM()
    vlm = vlm or GPT4V()
    detector = detector or OWLViT()
    llm = llm or GPT4()

    if agent_type == 'SegVLM':
        return SegVLM(segmentor=segmentor, vlm=vlm, configs=configs)

    elif agent_type == 'DetVLM':
        return DetVLM(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs)

    elif agent_type == 'DetLLM':
        return DetLLM(segmentor=segmentor, detector=detector, llm=llm, configs=configs)

    elif agent_type == 'VLMDet':
        return VLMDet(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs)

    elif agent_type == 'VLMDetInspect':
        return VLMDetInspect(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs)

    elif agent_type == 'DOM':
        return DOM(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs)
    
    else:
        raise ValueError("Unknown agent type.")