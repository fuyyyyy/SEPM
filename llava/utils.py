import datetime
import logging
import logging.handlers
import os
import sys
import json
import torch

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

def stage1_to_2(answer1_path, question1_path, question2_path):
    question_map = {}
    
    with open(answer1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line)
            question_id = data.get('question_id')
            text = data.get('text')
            mask_index = data.get('mask_index')  
            question_map[question_id] = {'text': text, 'mask_index': mask_index}

    output_data = []

    with open(question1_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            data = json.loads(line)
            question_id = data.get('question_id')
            current_text = data.get('text')

            if question_id in question_map:
                text_from_file1 = question_map[question_id]['text']
                mask_index_from_file1 = question_map[question_id]['mask_index']  

                if text_from_file1 == 'A':  
                    new_text = """Identify the predominant emotion conveyed.
Which of the following descriptions is represented in the image?
A. Amusement
C. Awe
D. Content
F. Excitement
Answer with the option's letter from the given choices directly."""
                elif text_from_file1 == 'B': 
                    new_text = """Identify the predominant emotion conveyed.
Which of the following descriptions is represented in the image?
B. Anger
E. Disgust
G. Fear
H. Sad
Answer with the option's letter from the given choices directly."""
                else:
                    string_parts = text_from_file1.split()
                    posi_rate = int(string_parts[0])
                    nega_rate = int(string_parts[1])

                    new_text = f"""Identify the predominant emotion conveyed.
This is an ambiguous emotion.
Which of the following descriptions is represented in the image?
A. Amusement
B. Anger
C. Awe
D. Content
E. Disgust
F. Excitement
G. Fear
H. Sad
Answer with the option's letter from the given choices directly."""

                data['text'] = new_text

                if mask_index_from_file1 is not None:
                    data['mask_index'] = mask_index_from_file1
                else:
                    data['mask_index'] = None

            output_data.append(data)

    with open(question2_path, 'w', encoding='utf-8') as out_file:
        for item in output_data:
            json.dump(item, out_file, ensure_ascii=False)
            out_file.write('\n')

    print(f'Processing is complete, and the modified data has been saved to {question2_path}.')
