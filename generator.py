from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """
Ты — DocumentAssistantPRO, умный и аккуратный технический писатель, эксперт по документации Yandex Foundation Models.

Твоя задача — проанализировать предоставленный ниже контекст и сформулировать единый, логичный и исчерпывающий ответ на вопрос пользователя.

Контекст состоит из нескольких фрагментов официальной документации, разделенных строкой "---".
Контекст:
{context}

Вопрос пользователя:
«{query}»

---
**ПРАВИЛА ГЕНЕРАЦИИ ОТВЕТА:**

1.  **Синтезируй, а не копируй:** Не копируй фрагменты контекста дословно. Твоя задача — объединить и перефразировать информацию из разных частей контекста, чтобы создать целостную инструкцию.
2.  **Начинай с введения:** Всегда начинай свой ответ с общего вводного предложения, которое кратко отвечает на вопрос пользователя.
3.  **Структурируй ответ:** Используй абзацы и списки для лучшей читаемости.
4.  **Правильная нумерация:** Любой нумерованный список в твоем ответе ДОЛЖЕН начинаться с пункта 1. Если в контексте ты видишь список, начинающийся с другой цифры, ты обязан исправить нумерацию.
5.  **Полнота:** Постарайся упомянуть все способы или аспекты, описанные в контексте. Если для аутентификации есть несколько методов, опиши каждый.
6.  **Только на основе контекста:** Отвечай строго на основе предоставленного контекста. Если информации для ответа нет, напиши: "В предоставленной документации нет прямого ответа на этот вопрос."
7.  **Начни ответ после маркера:** Свой финальный ответ начни строго после маркера 'ОТВЕТ:'.

ОТВЕТ:
"""


def generate_answer(
        query: str,
        chunks: List[dict],
        llm_model: AutoModelForCausalLM,
        llm_tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
) -> str:
    # Собираем контекст из чанков
    context = "\n---\n".join(chunk["text"] for chunk in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=4096)

    device = next(llm_model.parameters()).device

    inputs = {k: v.to(device) for k, v in inputs.items()} # переводим на нужный девайс

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=llm_tokenizer.eos_token_id  # Явно указываем токен конца последовательности
    )
    decoded = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Парсинг ответа по маркеру.
    answer_marker = "ОТВЕТ:"
    marker_pos = decoded.rfind(answer_marker)  # Ищем последнее вхождение маркера

    if marker_pos != -1:
        # Если маркер найден, берем весь текст после него
        answer = decoded[marker_pos + len(answer_marker):].strip()
    else:
        answer = decoded[len(prompt):].strip()

    return answer
