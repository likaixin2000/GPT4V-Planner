

import argparse
import datetime
import os
import time

import requests

import PIL

import openai

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

OPENAI_KEY = os.environ.get('OPENAI_API')







