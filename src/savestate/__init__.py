"""savestate: a small, self-maintaining state card on top of any agent memory system.

layer = Layer(Mem0Http("http://localhost:8888", user_id="me"), AnthropicLLM(), card_path="card.json")
layer.after_conversation(messages, date="2026-11-23")
prompt = layer.context("Where do I live now?")
"""

from savestate.card import Card
from savestate.layer import Layer
from savestate.llm import AnthropicLLM, OpenAICompatLLM
from savestate.models import LLM, Ingesting, Memory, Reranker, Store
from savestate.owners import Owners
from savestate.recall import Recall, answer_rules
from savestate.rerank import HttpReranker, LocalCrossEncoder, NoRerank
from savestate.stores import Honcho, InMemory, Letta, Mem0Http, MemOSHttp

__all__ = [
    "LLM",
    "AnthropicLLM",
    "Card",
    "Honcho",
    "HttpReranker",
    "InMemory",
    "Ingesting",
    "Layer",
    "Letta",
    "LocalCrossEncoder",
    "Mem0Http",
    "MemOSHttp",
    "Memory",
    "NoRerank",
    "OpenAICompatLLM",
    "Owners",
    "Recall",
    "Reranker",
    "Store",
    "answer_rules",
]
__version__ = "0.1.1"
