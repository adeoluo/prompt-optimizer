"""A bundled intent-classification task used for demonstrations and evaluation.

The task is to label short customer messages as ``refund``, ``shipping``, or
``cancel``. Messages are separable by keyword overlap so in-context learning has
a clear signal. The three splits are disjoint but share vocabulary, so a prompt
compiled on the training and validation splits generalizes to the test split.
"""

from __future__ import annotations

from prompt_optimizer.types import Example

NAIVE_INSTRUCTION = "Answer the customer message with a single word."

CANDIDATE_INSTRUCTIONS: tuple[str, ...] = (
    NAIVE_INSTRUCTION,
    "Read the customer message and respond.",
    "Classify the customer message. Valid labels: refund, shipping, cancel",
    "Categorize the message into exactly one label. Labels: refund, shipping, cancel",
)

TRAIN: tuple[Example, ...] = (
    Example("I want my money back please give me a refund", "refund"),
    Example("refund my payment I was overcharged on the order", "refund"),
    Example("I was charged twice I need a refund for the money", "refund"),
    Example("where is my package tracking says shipping is delayed", "shipping"),
    Example("what is the delivery status of my parcel", "shipping"),
    Example("the delivery is late when will my parcel arrive", "shipping"),
    Example("please cancel my order before it ships", "cancel"),
    Example("I want to cancel my recurring subscription plan", "cancel"),
    Example("stop my plan and cancel the subscription", "cancel"),
)

VALIDATION: tuple[Example, ...] = (
    Example("give me a refund for the money I was charged", "refund"),
    Example("where is my package the shipping is delayed", "shipping"),
    Example("what is the delivery status of my parcel order", "shipping"),
    Example("cancel my recurring subscription plan", "cancel"),
)

TEST: tuple[Example, ...] = (
    Example("I need my money back as a refund for the charge", "refund"),
    Example("refund the payment I was overcharged", "refund"),
    Example("my package shipping has not arrived and is delayed", "shipping"),
    Example("what is the delivery status of my late parcel", "shipping"),
    Example("please cancel my order and subscription plan", "cancel"),
    Example("I want to cancel the recurring plan", "cancel"),
)
