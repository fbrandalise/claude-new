"""
Evaluation pipeline using DeepEval to score enrichment quality.
"""

import json
import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def build_metrics(judge_model: str | None = None) -> list:
    """
    Returns a list of DeepEval metrics tailored for product attribute
    enrichment evaluation.
    """
    judge = judge_model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4.1-mini")

    completeness = GEval(
        name="Completude",
        criteria=(
            "Avalie se o 'actual output' extraiu TODOS os atributos presentes "
            "no 'expected output'. Penalize atributos faltantes. "
            "Um score alto significa que nenhum atributo relevante foi omitido."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=judge,
    )

    correctness = GEval(
        name="Corretude",
        criteria=(
            "Avalie se os VALORES dos atributos no 'actual output' estão "
            "corretos quando comparados com o 'expected output'. "
            "Penalize valores errados, imprecisos ou inventados. "
            "Um score alto significa alta precisão nos valores extraídos."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=judge,
    )

    consistency = GEval(
        name="Consistência de Formato",
        criteria=(
            "Avalie se o 'actual output' segue um formato JSON válido e "
            "consistente, com nomes de atributos em snake_case em português, "
            "valores bem formatados, e sem campos desnecessários ou redundantes. "
            "Verifique também se inclui 'categoria' e 'tipo_produto'."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=judge,
    )

    no_hallucination = GEval(
        name="Sem Alucinação",
        criteria=(
            "Avalie se o 'actual output' contém APENAS informações que estão "
            "presentes no 'input' (descrição do produto). "
            "Penalize severamente qualquer atributo ou valor que foi INVENTADO "
            "e não aparece na descrição original. "
            "Um score alto significa zero informações fabricadas."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=judge,
    )

    return [completeness, correctness, consistency, no_hallucination]


def create_test_case(
    product_name: str,
    raw_description: str,
    actual_output: dict | str,
    expected_attributes: dict,
) -> LLMTestCase:
    """Creates a DeepEval test case for one product enrichment."""
    if isinstance(actual_output, dict):
        actual_str = json.dumps(actual_output, ensure_ascii=False, indent=2)
    else:
        actual_str = actual_output

    expected_str = json.dumps(expected_attributes, ensure_ascii=False, indent=2)

    return LLMTestCase(
        input=f"Nome: {product_name}\nDescrição: {raw_description}",
        actual_output=actual_str,
        expected_output=expected_str,
    )


def run_evaluation(test_case: LLMTestCase, metrics: list) -> list[dict]:
    """
    Runs all metrics on a single test case and returns results.
    """
    results = []
    for metric in metrics:
        metric.measure(test_case)
        results.append({
            "metric": metric.name,
            "score": round(metric.score, 3),
            "passed": metric.is_successful(),
            "reason": metric.reason if hasattr(metric, "reason") else "",
        })
    return results
