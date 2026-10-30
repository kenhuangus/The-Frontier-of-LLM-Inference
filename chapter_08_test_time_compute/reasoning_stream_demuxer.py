"""
Reasoning Stream Demuxer
Separates internal thinking tokens (<think> ... </think>) from final user completions,
robustly handling split token boundaries, partial streaming tags, and multiple reasoning phases.
"""
from typing import Generator, Iterable, List, Optional, Tuple


class ReasoningStreamDemuxer:
    """
    Streaming token demultiplexer for test-time reasoning models (e.g., DeepSeek R1/V4, Kimi K3).
    Separates internal deliberation tokens (<think> ... </think>) from external response tokens.
    
    Robust against:
    - Token boundary fragmentation (e.g., '<th' followed by 'ink>')
    - Embedded tags within single token strings (e.g., '<think>Step 1')
    - Multiple sequential <think> blocks
    - Incomplete / truncated thinking streams (missing closing tag)
    - Empty thought sections (<think></think>)
    - Standard non-reasoning streams (pass-through)
    """

    START_TAG = "<think>"
    END_TAG = "</think>"

    def __init__(
        self,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
        in_think: bool = False,
    ):
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.in_think = in_think
        self.thinking_tokens: List[str] = []
        self.response_tokens: List[str] = []
        self._buffer: str = ""

        # Precompute candidate prefixes ordered from longest to shortest
        self._start_prefixes = [self.start_tag[:i] for i in range(len(self.start_tag) - 1, 0, -1)]
        self._end_prefixes = [self.end_tag[:i] for i in range(len(self.end_tag) - 1, 0, -1)]

    @property
    def is_thinking(self) -> bool:
        """Indicates whether the demuxer is currently inside a <think> reasoning block."""
        return self.in_think

    def reset(self, in_think: bool = False) -> None:
        """Resets the internal stream demuxer state."""
        self.in_think = in_think
        self.thinking_tokens.clear()
        self.response_tokens.clear()
        self._buffer = ""

    def feed_token(self, token: Optional[str]) -> None:
        """
        Processes a single streamed token chunk, updating thinking and response token lists.
        Handles partial delimiter prefixes and cross-boundary tags.
        """
        if not token:
            return

        text = self._buffer + token
        self._buffer = ""

        while text:
            if not self.in_think:
                idx = text.find(self.start_tag)
                if idx != -1:
                    pre = text[:idx]
                    if pre:
                        self.response_tokens.append(pre)
                    self.in_think = True
                    text = text[idx + len(self.start_tag):]
                else:
                    found_prefix = None
                    for p in self._start_prefixes:
                        if text.endswith(p):
                            found_prefix = p
                            break

                    if found_prefix:
                        pre = text[:-len(found_prefix)]
                        if pre:
                            self.response_tokens.append(pre)
                        self._buffer = found_prefix
                        text = ""
                    else:
                        self.response_tokens.append(text)
                        text = ""
            else:
                idx = text.find(self.end_tag)
                if idx != -1:
                    pre = text[:idx]
                    if pre:
                        self.thinking_tokens.append(pre)
                    self.in_think = False
                    text = text[idx + len(self.end_tag):]
                else:
                    found_prefix = None
                    for p in self._end_prefixes:
                        if text.endswith(p):
                            found_prefix = p
                            break

                    if found_prefix:
                        pre = text[:-len(found_prefix)]
                        if pre:
                            self.thinking_tokens.append(pre)
                        self._buffer = found_prefix
                        text = ""
                    else:
                        self.thinking_tokens.append(text)
                        text = ""

    def feed_tokens(self, tokens: Iterable[str]) -> None:
        """Feeds an iterable of token chunks sequentially."""
        for tok in tokens:
            self.feed_token(tok)

    def finalize(self) -> None:
        """
        Flushes any remaining buffered text at end of stream.
        Should be called when the generator stream finishes.
        """
        if self._buffer:
            if self.in_think:
                self.thinking_tokens.append(self._buffer)
            else:
                self.response_tokens.append(self._buffer)
            self._buffer = ""

    def get_thinking_text(self) -> str:
        """Returns the full reconstructed thinking text."""
        buffered = self._buffer if self.in_think else ""
        return "".join(self.thinking_tokens) + buffered

    def get_response_text(self) -> str:
        """Returns the full reconstructed final response text."""
        buffered = self._buffer if not self.in_think else ""
        return "".join(self.response_tokens) + buffered

    def has_thinking_content(self) -> bool:
        """Checks if any non-empty thinking deliberation was recorded."""
        return bool(self.get_thinking_text().strip())

    def feed_stream(self, stream: Iterable[str]) -> Generator[Tuple[str, str], None, None]:
        """
        Generator demuxing a stream in real-time.
        Yields (channel, text) tuples where channel is 'think' or 'response'.
        """
        for token in stream:
            prev_think_len = len(self.thinking_tokens)
            prev_resp_len = len(self.response_tokens)

            self.feed_token(token)

            # Yield newly emitted thinking tokens
            while prev_think_len < len(self.thinking_tokens):
                yield ("think", self.thinking_tokens[prev_think_len])
                prev_think_len += 1

            # Yield newly emitted response tokens
            while prev_resp_len < len(self.response_tokens):
                yield ("response", self.response_tokens[prev_resp_len])
                prev_resp_len += 1

        if self._buffer:
            channel = "think" if self.in_think else "response"
            buf = self._buffer
            self.finalize()
            yield (channel, buf)


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 8: REASONING STREAM DEMUXER")
    print("=" * 80)

    # Standard stream
    demuxer = ReasoningStreamDemuxer()
    stream = ["<think>", "Analyze", " memory", " wall", " constraint", "</think>", "The", " optimal", " solution", " is", " MLA."]
    for tok in stream:
        demuxer.feed_token(tok)
    demuxer.finalize()
    print("Standard Stream:")
    print("  Thinking Tokens Extracted:", "".join(demuxer.thinking_tokens))
    print("  Final Output Content    :", "".join(demuxer.response_tokens))

    # Partial / fragmented stream demonstration
    print("\nFragmented Token Stream ('<th' then 'ink>'):")
    fragmented_stream = ["<th", "ink>", "Evaluating", " compute", " roofline", " bounds", "</th", "ink>", "Result: 42"]
    frag_demuxer = ReasoningStreamDemuxer()
    for tok in fragmented_stream:
        frag_demuxer.feed_token(tok)
    frag_demuxer.finalize()
    print("  Thinking Tokens Extracted:", frag_demuxer.get_thinking_text())
    print("  Final Output Content    :", frag_demuxer.get_response_text())
    print("=" * 80)
