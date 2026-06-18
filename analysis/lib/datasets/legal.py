import json
import os
from typing import Optional


SYSTEM_PROMPT = \
f"""
You are a judge expert in Saudi law. Your task is to produce the reasoning, verdict, and classification of a legal case from Saudi Arabia.
The cases involve trade and finance and commercial laws.

## Task
You will be given a set of facts and laws from the case, and you MUST provide THREE sections:
1. A reasoning section analyzing the facts
2. A verdict prediction section stating a summary of what you think the court will decide
3. A classification of the verdict

## Guidelines:
- Provide a verdict based on the facts and the saudi laws
- Be assertive and clear in your verdict, as if you were a judge yourself
- Base the verdict only on the facts provided without personal opinions or biases
- Your verdict and reasoning should be strictly in english
- The reasoning should be detailed and step-by-step, each step separated by a period
- The verdict should be short and direct

## Classification options
Classification options (Choose only one):
- PLAINTIFF: Court ruled in favor of the plaintiff
- DEFENDANT: Court ruled in favor of the defendant
- DISMISSED: No ruling issued (dismissed, lack of jurisdiction, procedural rejection)
- SETTLEMENT: Parties reached settlement/reconciliation

## Format
Follow this exact format:

<REASONING>
Your detailed reasoning and analysis here. each step of your reasoning should be separated by a period.
</REASONING>

<VERDICT>
Your clear and direct verdict statement summary here
</VERDICT>

<VERDICT_CLASSIFICATION>
Your classification here
</VERDICT_CLASSIFICATION>


Do not output anything outside these three sections.
"""

ARABIC_SYSTEM_PROMPT = \
f"""
أنت قاضٍ خبير في النظام القانوني السعودي. مهمتك هي تقديم التحليل والحكم وتصنيف الحكم لقضية قانونية من المملكة العربية السعودية. تتعلق القضايا بالأنظمة التجارية والمالية والتجارية.

## Task
أنت قاضٍ خبير في النظام القانوني السعودي. مهمتك هي تقديم التحليل والحكم وتصنيف الحكم لقضية قانونية من المملكة العربية السعودية.
تتعلق القضايا بالأنظمة التجارية والمالية والتجارية.
سيتم تزويدك بمجموعة من الوقائع والأنظمة الخاصة بالقضية، ويجب عليك تقديم ثلاثة أقسام:
1. قسم التحليل والتعليل لتحليل الوقائع
2. قسم التنبؤ بالحكم يوضح ملخصاً لما تعتقد أن المحكمة ستقرره
3. تصنيف الحكم

## Guidelines:
- قدّم الحكم بناءً على الوقائع والأنظمة السعودية
- كن حازماً وواضحاً في حكمك، كما لو كنت قاضياً بنفسك
- استند في الحكم فقط إلى الوقائع المقدمة دون آراء أو تحيزات شخصية
- يجب أن يكون حكمك وتحليلك باللغة العربية حصراً
- يجب أن يكون التحليل مفصلاً وخطوة بخطوة، كل خطوة مفصولة بنقطة
- يجب أن يكون الحكم قصيراً ومباشراً

## Classification options
خيارات التصنيف (اختر خيارًا واحدًا فقط):
- المدعي: حكمت المحكمة لصالح المدعي
- المدعى عليه: حكمت المحكمة لصالح المدعى عليه
- رفض الدعوى: لم يصدر حكم في الموضوع (رفض الدعوى، عدم الاختصاص، أو رفض إجرائي)
- تسوية: توصل الأطراف إلى تسوية أو صلح

## Format
اتبع هذا التنسيق بالضبط:

<REASONING>
تحليلك المفصل وتعليلك هنا. يجب أن تكون كل خطوة من خطوات تحليلك مفصولة بنقطة.
</REASONING>

<VERDICT>
ملخص بيان حكمك الواضح والمباشر هنا
</VERDICT>

<VERDICT_CLASSIFICATION>
تصنيفك هنا
</VERDICT_CLASSIFICATION>

لا تُخرج أي شيء خارج هذه الأقسام الثلاثة. يجب أن تكون جميع ردودك باللغة العربية فقط.
"""


# Canonical English classification options.
CLASSIFICATION_OPTIONS = ["PLAINTIFF", "DEFENDANT", "DISMISSED", "SETTLEMENT"]

# Map Arabic classification labels (as written in ARABIC_SYSTEM_PROMPT) to
# their canonical English form. Used so an Arabic completion can be compared
# against the English ground-truth label during evaluation.
#
# IMPORTANT: keep keys in sync with the labels in ARABIC_SYSTEM_PROMPT's
# "## Classification options" section.
ARABIC_TO_ENGLISH_CLASSIFICATION = {
    "المدعي":      "PLAINTIFF",
    "المدعى عليه": "DEFENDANT",
    "رفض الدعوى":  "DISMISSED",
    "تسوية":       "SETTLEMENT",
}


def normalize_classification(label):
    """Normalize a classification string to its canonical English form.

    Accepts:
      - English label (PLAINTIFF / DEFENDANT / DISMISSED / SETTLEMENT,
        case-insensitive, possibly with extra surrounding text)
      - Arabic label (المدعي, المدعى عليه, رفض الدعوى, تسوية)
    Returns one of CLASSIFICATION_OPTIONS, or None if no match.
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None

    # Try Arabic first; check longer keys first so "المدعى عليه" wins over "المدعى".
    for ar, en in sorted(
        ARABIC_TO_ENGLISH_CLASSIFICATION.items(), key=lambda p: -len(p[0])
    ):
        if ar in s:
            return en

    # English (substring match, case-insensitive)
    su = s.upper()
    for opt in CLASSIFICATION_OPTIONS:
        if opt in su:
            return opt

    return None


class LegalDataset:
    """
    Dataset for legal cases loaded from a JSON file.

    Each sample in the JSON is a dict with:
        - facts:                  list of strings (individual fact statements)
        - laws:                   list of strings (applicable laws)
        - steps_texts:            list of strings (reasoning steps)
        - verdict_classification: str (English or Arabic) — GT label
    """

    def __init__(
        self,
        json_path: str = "/home/sabeasm/llm_effective_depth/data/single_arabic_case.json",
        max_samples: Optional[int] = None,
        language: str = "english",                 # ← NEW: "english" or "arabic"
        facts_key: str = "facts",
        laws_key: str = "laws",
        steps_key: str = "steps_texts",
        verdict_classification_key: str = "verdict_classification",
        facts_separator: str = " ",
        laws_separator: str = " ",
        steps_separator: str = " ",
        include_steps: bool = False,
        use_chat_template: bool = True,
    ):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        lang = language.lower().strip()
        if lang not in ("english", "arabic"):
            raise ValueError(
                f"Unsupported language {language!r}. Use 'english' or 'arabic'."
            )

        self.json_path = json_path
        self.max_samples = max_samples
        self.language = lang
        self.facts_key = facts_key
        self.laws_key = laws_key
        self.steps_key = steps_key
        self.verdict_classification_key = verdict_classification_key
        self.facts_separator = facts_separator
        self.laws_separator = laws_separator
        self.steps_separator = steps_separator
        self.include_steps = include_steps
        self.use_chat_template = use_chat_template

        self.data = self._load(json_path)

    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        return ARABIC_SYSTEM_PROMPT if self.language == "arabic" else SYSTEM_PROMPT

    def _load(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a JSON list at the top level, got {type(raw)}"
            )
        if self.max_samples is not None:
            raw = raw[: self.max_samples]
        return raw

    def _join(self, lst, separator):
        if lst is None:
            return ""
        return separator.join(s for s in lst if s and s.strip())

    def _build_user_message(self, sample: dict) -> str:
        facts = self._join(sample.get(self.facts_key, []), self.facts_separator)
        laws  = self._join(sample.get(self.laws_key,  []), self.laws_separator)

        user_message = (
            f"## Facts\n{facts}\n\n"
            f"## Applicable Laws\n{laws}\n"
        )

        if self.include_steps:
            steps = self._join(sample.get(self.steps_key, []), self.steps_separator)
            if steps:
                user_message += f"\n## Reasoning Steps\n{steps}\n"

        return user_message

    # ------------------------------------------------------------------
    # Ground-truth accessors — always return canonical English
    # ------------------------------------------------------------------

    def get_verdict_classification(self, idx: int) -> Optional[str]:
        """Return the ground-truth label for sample `idx`, normalized to
        canonical English (so it can be compared against any prediction
        regardless of the prediction's language)."""
        v = self.data[idx].get(self.verdict_classification_key, None)
        return normalize_classification(v)

    def get_all_verdict_classifications(self) -> list:
        return [self.get_verdict_classification(i) for i in range(len(self.data))]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def format_sample(self, sample: dict) -> str:
        system_prompt = self._get_system_prompt()
        user_message  = self._build_user_message(sample)

        if self.use_chat_template:
            return (
                f"<system>\n{system_prompt.strip()}\n</system>\n\n"
                f"<user>\n{user_message.strip()}\n</user>\n\n"
                f"<assistant>"
            )
        return f"{system_prompt.strip()}\n\n{user_message.strip()}"

    def format_sample_for_classification_probe(self, sample_or_idx) -> str:
        if isinstance(sample_or_idx, int):
            sample = self.data[sample_or_idx]
        else:
            sample = sample_or_idx

        system_prompt = self._get_system_prompt()
        user_message  = self._build_user_message(sample)

        if self.use_chat_template:
            return (
                f"<system>\n{system_prompt.strip()}\n</system>\n\n"
                f"<user>\n{user_message.strip()}\n</user>\n\n"
                f"<assistant>\n"
                f"<VERDICT_CLASSIFICATION>\n"
            )
        return (
            f"{system_prompt.strip()}\n\n"
            f"{user_message.strip()}\n\n"
            f"<VERDICT_CLASSIFICATION>\n"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            yield self.format_sample(sample)

    def __getitem__(self, idx):
        return self.format_sample(self.data[idx])