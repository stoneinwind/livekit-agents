class VisionTexTProcessor:
    def __init__(self):
        self.last_output = ""

    def _should_speak(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False

        # 去重
        if text == self.last_output:
            return False

        self.last_output = text
        return True

    def process(self, text: str):
        if self._should_speak(text):
            return text
        return None