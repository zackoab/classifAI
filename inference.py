from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm

from prompts import (
    classification_prompt,
    test_dataset_generation_prompt,
)


class ClassificationResponse(BaseModel):
    category: str
    reason: str


def predict_class(
    llm_model: str,
    concept: str,
    class_1: str,
    class_2: str,
    verbatim: str,
) -> ClassificationResponse | None:
    PROMPT = classification_prompt(concept=concept, classe_1=class_1, classe_2=class_2)
    client = OpenAI(api_key=f"{os.getenv("OPENAI_API_KEY")}")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Verbatim : {verbatim}\nClasse :"},
    ]
    response = client.beta.chat.completions.parse(
        model=llm_model,
        messages=messages,  # type: ignore
        response_format=ClassificationResponse,
        max_tokens=200,
        temperature=0,
    )
    content = response.choices[0].message.parsed
    return content


def batch_inference(
    model_parameters: dict[str, str], inputs: list[str]
) -> list[tuple[str, str, str]]:
    predictions = []
    with ThreadPoolExecutor() as executor:
        future_to_row = {
            executor.submit(predict_class, **model_parameters, verbatim=row): row
            for row in inputs
        }
        for future in tqdm(
            as_completed(future_to_row), total=len(inputs), desc="Evaluating..."
        ):
            prediction = future.result().category  # type: ignore
            reason = future.result().reason  # type: ignore
            predictions.append((future_to_row[future], prediction, reason))  # type: ignore
    return predictions  # type: ignore


def generate_test_dataset(
    llm_model: str, concept: str, class_1: str, class_2: str, num_samples: int
) -> str | None:
    """Generates a synthetic dataset for testing."""
    print(f"Generating test dataset with {num_samples} verbatim...")
    PROMPT = test_dataset_generation_prompt(
        concept=concept, class_1=class_1, class_2=class_2, num_samples=num_samples
    )
    client = OpenAI(api_key=f"{os.getenv("OPENAI_API_KEY")}")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Dataset :"},
    ]
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,  # type: ignore
        temperature=1,
    )
    content = response.choices[0].message.content
    print(f"Dataset generated.")
    return content
