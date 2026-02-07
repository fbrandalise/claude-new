"""
Two prompt variations for product attribute enrichment.
Each prompt takes a product name + raw description and should return
structured JSON with extracted attributes.
"""

PROMPT_A = """Você é um especialista em catalogação de produtos para e-commerce.

Dado o nome e a descrição de um produto, extraia todos os atributos técnicos relevantes
e retorne um JSON estruturado.

## Regras:
- Extraia TODOS os atributos mencionados na descrição
- Use nomes de atributo em português, em snake_case
- Sempre inclua "categoria" e "tipo_produto"
- Se um atributo não estiver presente na descrição, NÃO o invente
- Retorne APENAS o JSON, sem explicações adicionais

## Produto:
Nome: {product_name}
Descrição: {raw_description}

## JSON com atributos extraídos:
"""

PROMPT_B = """Você é um analista de dados de produto trabalhando para um grande marketplace.

Sua tarefa é criar uma ficha técnica completa a partir da descrição fornecida.

## Instruções detalhadas:
1. Leia atentamente o nome e a descrição do produto
2. Identifique a categoria e tipo do produto
3. Extraia cada especificação técnica mencionada
4. Organize os atributos de forma hierárquica e padronizada
5. Use nomes de atributo em português, formato snake_case
6. Inclua campos "categoria" e "tipo_produto" baseados no contexto
7. Para atributos booleanos, use "Sim" ou "Não"
8. Mantenha as unidades de medida originais (kg, mm, W, etc.)

## IMPORTANTE:
- NÃO invente informações que não estejam na descrição
- NÃO adicione atributos genéricos que não agregam valor
- Retorne SOMENTE o JSON válido, sem markdown ou explicações

## Produto para análise:
Nome: {product_name}
Descrição: {raw_description}

## Ficha técnica (JSON):
"""

PROMPTS = {
    "Prompt A - Direto e conciso": PROMPT_A,
    "Prompt B - Detalhado com instruções passo-a-passo": PROMPT_B,
}
