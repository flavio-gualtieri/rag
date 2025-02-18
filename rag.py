import logging
import random
import re

class RAG:
    def __init__(self, model, chunks = 20, window = 10):
        self.chunks = chunks
        self.window = window
        self.model = model
        self.logger = self.__init_logger()

    def __init_logger(_self):
        return setup_logger("RAG_LOGGER", level = logging.INFO, console = True, file = False)

    def __create_prompt(self, question, book):
        pass

    