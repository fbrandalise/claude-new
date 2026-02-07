"""
Streamlit app for comparing two prompt variations for product attribute
enrichment, evaluated with DeepEval.
"""

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.enrichment import enrich_product
from src.evaluator import build_metrics, create_test_case, run_evaluation
from src.prompts import PROMPTS

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Evaluator - Enriquecimento de Produtos",
    page_icon="📊",
    layout="wide",
)

st.title("Avaliador de Prompts - Enriquecimento de Atributos de Produto")
st.caption(
    "POC usando DeepEval para comparar a qualidade de duas variações de prompt "
    "que extraem fichas técnicas de produtos."
)

# ---------------------------------------------------------------------------
# Sidebar - Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuração")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Necessária para o LLM de enriquecimento e para o juiz do DeepEval",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    enrichment_model = st.selectbox(
        "Modelo de Enriquecimento",
        ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        index=0,
    )

    judge_model = st.selectbox(
        "Modelo Juiz (DeepEval)",
        ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        index=0,
    )

    st.divider()
    st.markdown(
        "**Como funciona:**\n"
        "1. Selecione um produto\n"
        "2. O sistema enriquece com cada prompt\n"
        "3. DeepEval avalia ambos os resultados\n"
        "4. Compare métricas lado a lado"
    )

# ---------------------------------------------------------------------------
# Load product data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "products.json")


@st.cache_data
def load_products() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


products = load_products()
product_names = {p["id"]: p["name"] for p in products}

# ---------------------------------------------------------------------------
# Prompt editing
# ---------------------------------------------------------------------------
st.header("1. Prompts")

prompt_keys = list(PROMPTS.keys())
col_pa, col_pb = st.columns(2)

with col_pa:
    st.subheader(prompt_keys[0])
    prompt_a_text = st.text_area(
        "Edite o Prompt A",
        value=PROMPTS[prompt_keys[0]],
        height=250,
        key="prompt_a",
    )

with col_pb:
    st.subheader(prompt_keys[1])
    prompt_b_text = st.text_area(
        "Edite o Prompt B",
        value=PROMPTS[prompt_keys[1]],
        height=250,
        key="prompt_b",
    )

# ---------------------------------------------------------------------------
# Product selection
# ---------------------------------------------------------------------------
st.header("2. Selecione os Produtos")

selected_ids = st.multiselect(
    "Produtos para avaliar",
    options=list(product_names.keys()),
    default=[products[0]["id"]],
    format_func=lambda x: f"{x} - {product_names[x]}",
)

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
st.header("3. Executar Avaliação")

if not api_key:
    st.warning("Configure sua OpenAI API Key na barra lateral para continuar.")
    st.stop()

if not selected_ids:
    st.info("Selecione pelo menos um produto acima.")
    st.stop()

if st.button("Executar Enriquecimento + Avaliação", type="primary", use_container_width=True):
    metrics = build_metrics(judge_model=judge_model)

    all_results = []

    progress = st.progress(0)
    total_steps = len(selected_ids) * 2  # 2 prompts per product
    step = 0

    for product in products:
        if product["id"] not in selected_ids:
            continue

        st.divider()
        st.subheader(f"Produto: {product['name']}")

        col_a, col_b = st.columns(2)

        for col, prompt_label, prompt_text in [
            (col_a, prompt_keys[0], prompt_a_text),
            (col_b, prompt_keys[1], prompt_b_text),
        ]:
            with col:
                st.markdown(f"**{prompt_label}**")

                with st.spinner("Enriquecendo atributos..."):
                    enriched, raw_output = enrich_product(
                        product_name=product["name"],
                        raw_description=product["raw_description"],
                        prompt_template=prompt_text,
                        model=enrichment_model,
                    )

                if enriched.get("_parse_error"):
                    st.error("Erro ao parsear JSON do modelo")
                    st.code(raw_output, language="text")
                    step += 1
                    progress.progress(step / total_steps)
                    continue

                with st.expander("Atributos Extraídos", expanded=True):
                    st.json(enriched)

                with st.spinner("Avaliando com DeepEval..."):
                    test_case = create_test_case(
                        product_name=product["name"],
                        raw_description=product["raw_description"],
                        actual_output=enriched,
                        expected_attributes=product["expected_attributes"],
                    )
                    eval_results = run_evaluation(test_case, metrics)

                # Display scores
                score_cols = st.columns(len(eval_results))
                for sc, res in zip(score_cols, eval_results):
                    with sc:
                        passed_icon = "✅" if res["passed"] else "❌"
                        st.metric(
                            label=f"{res['metric']}",
                            value=f"{res['score']:.1%}",
                            help=res["reason"],
                        )
                        st.caption(passed_icon)

                for res in eval_results:
                    all_results.append({
                        "produto": product["id"],
                        "prompt": prompt_label,
                        "metrica": res["metric"],
                        "score": res["score"],
                        "passou": res["passed"],
                        "razao": res["reason"],
                    })

            step += 1
            progress.progress(step / total_steps)

    progress.progress(1.0)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    if all_results:
        st.divider()
        st.header("4. Resumo Comparativo")

        df = pd.DataFrame(all_results)

        pivot = df.pivot_table(
            index=["produto", "metrica"],
            columns="prompt",
            values="score",
            aggfunc="first",
        ).reset_index()

        st.dataframe(pivot, use_container_width=True)

        # Averages per prompt
        st.subheader("Média por Prompt")
        avg_df = df.groupby("prompt")["score"].mean().reset_index()
        avg_df.columns = ["Prompt", "Score Médio"]
        avg_df["Score Médio"] = avg_df["Score Médio"].apply(lambda x: f"{x:.1%}")

        st.dataframe(avg_df, use_container_width=True, hide_index=True)

        # Averages per metric
        st.subheader("Média por Métrica")
        metric_avg = df.pivot_table(
            index="metrica",
            columns="prompt",
            values="score",
            aggfunc="mean",
        ).reset_index()

        st.dataframe(metric_avg, use_container_width=True, hide_index=True)

        # Winner
        st.subheader("Vencedor")
        prompt_avgs = df.groupby("prompt")["score"].mean()
        winner = prompt_avgs.idxmax()
        winner_score = prompt_avgs.max()
        loser = prompt_avgs.idxmin()
        loser_score = prompt_avgs.min()

        delta = winner_score - loser_score
        st.success(
            f"**{winner}** venceu com score médio de **{winner_score:.1%}** "
            f"(+{delta:.1%} vs {loser} com {loser_score:.1%})"
        )
