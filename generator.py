from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "Answer the user question based on the following context:\n\n"
    "{context}\n\n"
    "Question: {query}\n\nAnswer:"
)

def generate_answer(
    query: str,
    chunks: List[dict],
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    device,
    max_new_tokens: int = 256,
) -> str:
    # Собираем контекст из чанков
    context = "\n---\n".join(chunk["text"] for chunk in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, device=device) # TODO протестировать перенос токенов на GPU
    outputs = llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Обрезаем исходный prompt
    answer = decoded[len(prompt):].strip()
    return answer
