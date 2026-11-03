"""
Safety Agent for Traffic Sign Recognition System
"""
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

from models import TrafficSignCNN

load_dotenv()

IMG_SIZE = 32
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.15
SAFETY_CRITICAL_THRESHOLD = 0.90
SAFETY_CRITICAL_CLASSES = {
    "Stop", "Yield", "No entry", "Right-of-way at the next intersection",
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "Speed limit (100km/h)", "Speed limit (120km/h)",
}


class ClassifierTool:
    def __init__(self, checkpoint_path: pathlib.Path, class_names: list, device: Optional[str] = None):
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.class_names = class_names
        self.model = TrafficSignCNN(num_classes=len(class_names))
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def classify(self, image: Image.Image) -> dict:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1).squeeze(0).cpu()
        topk = torch.topk(probs, k=3)
        top3 = [[self.class_names[i], float(p)] for p, i in zip(topk.values, topk.indices)]
        return {
            "predicted_class": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
            "margin": top3[0][1] - top3[1][1],
        }


@dataclass
class FlaggedCase:
    case_id: str
    predicted_class: str
    confidence: float
    margin: float
    reasons: list
    decision: str
    llm_rationale: str
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    def __init__(self):
        self.cases: list = []

    def record(self, case: FlaggedCase) -> None:
        self.cases.append(case)

    def recurrence_count(self, predicted_class: str) -> int:
        return sum(1 for c in self.cases if c.predicted_class == predicted_class)

    def is_recurring(self, predicted_class: str, threshold: int = 2) -> bool:
        return self.recurrence_count(predicted_class) >= threshold


PERSONA = """You are an ADAS Safety Triage Agent. You review one traffic-sign \
classification made by a CNN perception model and decide whether it is safe to accept \
automatically or whether it must be escalated to a human reviewer. You never see the \
raw image, only the classifier's structured output and the result of deterministic \
safety checks that already ran before you. If the deterministic check already forced \
ESCALATE, that decision is final: your only job is to write a one-to-two sentence \
rationale for the human reviewer's log explaining why. If no deterministic check fired, \
you may still recommend ESCALATE on your own judgement (for example if the top-2 class \
is also safety-critical, or the session memory shows this class keeps recurring), but \
you must never recommend ACCEPT when a deterministic check already forced ESCALATE. \
Always reply with strict JSON only: {"decision": "ACCEPT" or "ESCALATE", "rationale": "..."}"""


class SafetyAgent:
    def __init__(self, classifier: ClassifierTool, memory: Optional[SessionMemory] = None,
                 client: Optional[anthropic.Anthropic] = None, model: str = "claude-haiku-5-5"):
        self.classifier = classifier
        self.memory = memory or SessionMemory()
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.log: list = []

    def _deterministic_safeguards(self, result: dict) -> list:
        reasons = []
        if result["confidence"] < CONFIDENCE_THRESHOLD:
            reasons.append(f"confidence {result['confidence']:.2f} below threshold {CONFIDENCE_THRESHOLD}")
        if result["margin"] < MARGIN_THRESHOLD:
            reasons.append(f"top-1/top-2 margin {result['margin']:.2f} below threshold {MARGIN_THRESHOLD} (ambiguous)")
        if (result["predicted_class"] in SAFETY_CRITICAL_CLASSES
                and result["confidence"] < SAFETY_CRITICAL_THRESHOLD):
            reasons.append(
                f"'{result['predicted_class']}' is safety-critical and requires confidence "
                f">= {SAFETY_CRITICAL_THRESHOLD}, got {result['confidence']:.2f}"
            )
        return reasons

    def _ask_llm(self, result: dict, forced_reasons: list, recurring: bool) -> dict:
        memory_note = (
            f"'{result['predicted_class']}' has already been escalated "
            f"{self.memory.recurrence_count(result['predicted_class'])} time(s) earlier this session."
            if recurring else "No prior escalations for this class in this session."
        )
        forced = ("ESCALATE (forced by deterministic safety checks): " + "; ".join(forced_reasons)
                  if forced_reasons else "none")
        user_msg = (
            f"Classifier output:\n"
            f"- predicted_class: {result['predicted_class']}\n"
            f"- confidence: {result['confidence']:.3f}\n"
            f"- top-3: {result['top3']}\n"
            f"- margin (top1-top2): {result['margin']:.3f}\n\n"
            f"Deterministic safety check result: {forced}\n"
            f"Session memory: {memory_note}\n\n"
            'Respond with strict JSON only, no other text: '
            '{"decision": "ACCEPT" or "ESCALATE", "rationale": "one short sentence"}'
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=500, system=PERSONA,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"decision": "ESCALATE",
                     "rationale": f"Could not parse LLM response as JSON, defaulting to escalate. Raw: {text[:200]}"}

    def review(self, image: Image.Image, case_id: str, true_label: str = None) -> dict:
        result = self.classifier.classify(image)
        forced_reasons = self._deterministic_safeguards(result)
        recurring = self.memory.is_recurring(result["predicted_class"])

        llm_response = self._ask_llm(result, forced_reasons, recurring)
        llm_decision = llm_response.get("decision", "ESCALATE")
        rationale = llm_response.get("rationale", "")

        if forced_reasons:
            final_decision = "ESCALATE"
            llm_overridden = llm_decision != "ESCALATE"
        else:
            final_decision = llm_decision
            llm_overridden = False

        if final_decision == "ESCALATE":
            self.memory.record(FlaggedCase(
                case_id=case_id, predicted_class=result["predicted_class"],
                confidence=result["confidence"], margin=result["margin"],
                reasons=forced_reasons, decision=final_decision, llm_rationale=rationale,
            ))

        entry = {
            "case_id": case_id, "true_label": true_label,
            "classifier_output": result, "deterministic_reasons": forced_reasons,
            "recurring": recurring, "llm_decision": llm_decision, "llm_rationale": rationale,
            "final_decision": final_decision, "llm_overridden": llm_overridden,
        }
        self.log.append(entry)
        return entry
