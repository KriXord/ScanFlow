from __future__ import annotations

from typing import Any, Sequence


DEFAULT_PROMPT_TEMPLATE = (
    "<image>\n"
    "Answer the question using only the information visible in the image. "
    "Return only the final short answer with no explanation.\n"
    "Question: {question}\n"
    "Answer:"
)

# DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE = (
#     "<image>\n"
#     "Carefully examine this chart and determine whether the following user assertion about the chart are correct.\n"
#     "User assertion: {question}.\n"
#     "Let's thinking the following qustions one by one first:\n"
#     "1. What is user's assertion?\n"
#     "2. What are queried entities?\n"
#     "3. What are corosponding color / line style / legend / ... for these entities?\n"
#     "4. What is this chart type? if it is bar / line / scatter plot, please notice its cordinate / ticks ...\n"
#     "5. What are the entities value?\n"
#     "6. What are entities ralationship?\n"
#     "Combined with your answers, please provide a simple 'Yes' or 'No' response without any additional content.\n"
#     "Your Answer:\n"
# )

# ic
DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE = (
    "You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [yes] or [no].\n"
    "Here is an example:\n"
    "User: <image>\n"
    "User: The figure is a line chart. Please answer yes or no.\n"
    "You: yes.\n"
    "\n"
    "Following the above example:\n"
    "The query from the User is: {question} Please answer yes or no.\n"
    "Your Answer:"
)

# v4
# DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE = (
#     "<image>\n"
#     "You are a data analyst, good at dealing with chart data. Now you are required to analyze a chart for the User. You only need to answer [yes] or [no].\n"
#     "Here is an example:\"\n"
#     "User: <image>\n"
#     "User: The figure is a line chart. Please answer yes or no.\n"
#     "You: yes.\n"
#     "\n"
#     "Following the above example:\n"
#     "The query from the User is: {question} Please answer yes or no.\n"
#     "Your Answer:"
# )

# v5
# DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE = (
#     "<image>\n"
#     "Question: {question}. Please answer no or yes. Answer:"
# )

DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are reading one exact value from the chart.\n"
    "Locate the exact series/category and x-axis position asked in the question.\n"
    "Return the corresponding y-axis value only.\n"
    "If the question asks for a percentage, output the value with the % symbol.\n"
    "Otherwise output only the number.\n"
    "Preserve decimal places from the chart when possible.\n"
    "Do not output units other than % when the question asks for a percentage.\n"
    "Do not output words.\n"
    "Do not output the x-axis label, month index, quarter name, or any explanation.\n"
    "Question: {question}\n"
    "Answer:"
)

_CHARTQAPRO_MULTI_ANSWER_SUFFIX = (
    "If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].\n"
    "Remember to generate the final answer only without any additional text!\n"
)

_CHARTQAPRO_SINGLE_WORD_NUMBER_PHRASE_GUIDANCE = (
    "Your answer should be a single word, number, or phrase. "
    "If the question is unanswerable based on the information in the provided image, "
    "your answer should be unanswerable. Do not generate units. "
    "But if numerical units such as million, m, billion, B, or K are required, "
    "use the exact notation shown in the chart.\n"
)

DEFAULT_CHARTQAPRO_FACTOID_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given a factoid question that you need to answer based on the provided image.\n"
    + _CHARTQAPRO_SINGLE_WORD_NUMBER_PHRASE_GUIDANCE
    + _CHARTQAPRO_MULTI_ANSWER_SUFFIX
    + "Question: {question}\n"
    "Answer:"
)

DEFAULT_CHARTQAPRO_MULTI_CHOICE_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given a question along with different possible answers. "
    "You need to select the correct answer from the provided image.\n"
    "Your answer should be one of the option letters only: a, b, c or d "
    "(just the letter itself without any additional text). "
    "If the question is unanswerable based on the information in the provided image, "
    "your answer should be unanswerable.\n"
    + _CHARTQAPRO_MULTI_ANSWER_SUFFIX
    + "Question: {question}\n"
    "Answer:"
)

DEFAULT_CHARTQAPRO_HYPOTHETICAL_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given a hypothetical question that you need to answer based on the provided image.\n"
    + _CHARTQAPRO_SINGLE_WORD_NUMBER_PHRASE_GUIDANCE
    + _CHARTQAPRO_MULTI_ANSWER_SUFFIX
    + "Question: {question}\n"
    "Answer:"
)

DEFAULT_CHARTQAPRO_FACT_CHECKING_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given a fact statement that you need to assess based on the provided image.\n"
    "Your answer should be either true or false (without any additional text). "
    "If the question is unanswerable based on the information in the provided image, "
    "your answer should be unanswerable.\n"
    + _CHARTQAPRO_MULTI_ANSWER_SUFFIX
    + "Question: {question}\n"
    "Answer:"
)

DEFAULT_CHARTQAPRO_CONVERSATIONAL_PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given a multi-turn conversation, and your job is to answer the final question "
    "based on the conversation history and the information in the provided image.\n"
    + _CHARTQAPRO_SINGLE_WORD_NUMBER_PHRASE_GUIDANCE
    + _CHARTQAPRO_MULTI_ANSWER_SUFFIX
    + "Conversation:\n"
    "{conversation}\n"
    "Question: {question}\n"
    "Answer:"
)


def normalize_chartqapro_question_type(question_type: Any) -> str:
    normalized = " ".join(str(question_type or "").strip().lower().split())
    aliases = {
        "factoid": "factoid",
        "multi choice": "multi choice",
        "multiple choice": "multi choice",
        "mcq": "multi choice",
        "hypothetical": "hypothetical",
        "fact checking": "fact checking",
        "fact-checking": "fact checking",
        "fact check": "fact checking",
        "conversational": "conversational",
        "conversation": "conversational",
    }
    return aliases.get(normalized, normalized)


def build_chartqapro_conversation_history(
    questions: Sequence[str],
    previous_predictions: Sequence[str | None],
    question_idx: int,
) -> str:
    if question_idx <= 0:
        return "None"

    turns: list[str] = []
    for idx in range(question_idx):
        question = questions[idx].strip() if idx < len(questions) else ""
        prediction = previous_predictions[idx].strip() if idx < len(previous_predictions) and previous_predictions[idx] else ""
        if question:
            turns.append(f"Q{idx + 1}: {question}")
        if prediction:
            turns.append(f"A{idx + 1}: {prediction}")
    return "\n".join(turns) if turns else "None"


def is_chartbench_value_question(question: str) -> bool:
    question_text = question.strip()
    if question_text.endswith("?"):
        return True

    question_lower = question_text.lower()
    return question_lower.startswith("according to this chart, what is")


def is_chartbench_percentage_question(question: str) -> bool:
    question_lower = question.strip().lower()
    return "percentage" in question_lower or "percent" in question_lower or "%" in question_lower
