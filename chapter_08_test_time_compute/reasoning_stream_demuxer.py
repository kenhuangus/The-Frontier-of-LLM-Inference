"""
Reasoning Stream Demuxer
Separates internal thinking tokens (<think> ... </think>) from final user completions.
"""
from typing import List, Dict

class ReasoningStreamDemuxer:
    def __init__(self):
        self.in_think = False
        self.thinking_tokens = []
        self.response_tokens = []

    def feed_token(self, token: str):
        if token == "<think>":
            self.in_think = True
            return
        elif token == "</think>":
            self.in_think = False
            return
            
        if self.in_think:
            self.thinking_tokens.append(token)
        else:
            self.response_tokens.append(token)

if __name__ == "__main__":
    demuxer = ReasoningStreamDemuxer()
    stream = ["<think>", "Analyze", " memory", " wall", " constraint", "</think>", "The", " optimal", " solution", " is", " MLA."]
    for tok in stream:
        demuxer.feed_token(tok)
    print("Thinking Tokens Extracted:", "".join(demuxer.thinking_tokens))
    print("Final Output Content    :", "".join(demuxer.response_tokens))
