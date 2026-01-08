import unittest
from transformers import AutoTokenizer
from lib.Template.BaseTemplate import ProcessorPool


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen3-1.7B")
    string = """
<Think> The initial Query Search on the original query has already been executed but returned no documents, indicating potential issues with query precision or data availability. Since no documents have been retrieved yet, a direct search might not have yielded results due to ambiguity or specificity. To better target the search, the query should be rephrased for clarity and precision. Using Query Rewrite will generate alternative versions of the query that might better align with the available documents. This step is crucial to ensure that the search captures the correct information without redundant attempts. </Think> <Action> Target 0 | Query Rewrite </Action><END>
""".strip()
    tag_config = {
        "Think": (
            [tokenizer.encode("<Think>"), tokenizer.encode(" <Think>"), tokenizer.encode("<Think> ")],    # 开始标签变体
            [tokenizer.encode("</Think>"), tokenizer.encode(" </Think>"), tokenizer.encode("</Think> "), tokenizer.encode("</Think><END>")]   # 结束标签变体
        ),
        "Action": (
            [tokenizer.encode("<Action>"), tokenizer.encode(" <Action>"), tokenizer.encode("<Action> ")],
            [tokenizer.encode("</Action>"), tokenizer.encode(" </Action>"), tokenizer.encode("</Action> "), 
             tokenizer.encode("</Action><END>"), tokenizer.encode(" </Action><END>"), tokenizer.encode("</Action><END> ")]
        )
    }

    processor = ProcessorPool(tag_config = tag_config, num_threads = 2, queue_timeout = 0.1)
    input_ids = tokenizer.encode(string)
    result = processor.process([input_ids])
    print(result)

    for tag in result[0]:
        print("=" * 20)
        print(tag["tag_name"])
        print(tokenizer.decode(tag["content"][0]['content']))

test_string = f"""
**You are a decision-making agent that can freely use the following tools to optimize document retrieval and ensure only the most relevant, high-quality results are included in your final output. The final response will be derived directly from the processed document list, so rigorous filtering, sorting, and refinement are critical.**

**Guideline:**
- Only high revelant document matters: Immediately reject any document that doesn't directly address the core query with clear evidence. Never include borderline-relevance documents—aim for 3-5 documents with 90%+ relevance confidence.
- Always conduct thorough pre-action analysis before using any tool. **Any action without analysis will be ignored. ** Explicitly state your reasoning in think section, including why the current approach is insufficient, how the chosen tool resolves this, and expected outcome. Never proceed without this step.
- When standard retrieval (e.g., [Query Search]) fails to yield higher-relevance documents, proactively use [Query Rewrite] or via API to generate new search angles. This is the critical escalation path when relevance plateaus—do not stop at the first 5 results.
- **Limit actions to exactly 3 per interaction; Limit interactions to exactly 4 per user's input; any actions beyond the third will be automatically ignored.**
- Action History is given on each input. Check the action history and avoid repeating actions as they are less valuable. 

**Tools:**
Basic:
1. [Query Search]: Retrieve exactly **10** documents from the target query. 
    - Note: Using the same query multiple times will return identical documents. To retrieve documents from different angles, first use [Query Rewrite] to generate refined queries.
2. [Delete Documents]: Delete target documents directly when document is low relevance. Supports multiple document indices. 
3. [Sort Documents]: Rearrange the document list by specifying the desired order of documents.
4. [Delete Query]: Delete irrelevant target queries from the query list. 
5. [Stop]: Terminate interaction when user input is clear enough for final answer. This provides the last opportunity to refine output before termination.

Advanced:
6. [Query Rewrite]: Generate multiple refined queries for the use inputs using LLM API and retrieved 3 document for each query. 
7. [Query Extract]: Generate queries from all documents for answering use's input based on LLM API and retrieved 3 document for each query. 
8. [Document Filter]: Delete all irrelevant documents from list by LLM API. 
    - Note: **Auto activate [Document Filter] after every step.**
9. [Summarize Documents]: Generate a concise summary confirming whether the documents collectively answer the query. If the summary confirms a direct or indirect answer, delete all source documents and append the summary as a new document.

**Input Format Specification**
- Action History: Chronological list of previous actions.
    - Format: [Interaction ID | Action: Action type | Target ID: i, j]
- Query Collected: All queries with status.
    - Format: <Query 0> Query Context... <Query 1> Query Context...
- Documents: Retrieved documents with title and context.
    - Format: <Document 0> Title... <Document 1> Title...
- User Input: Original user query.

**Response Format Specification**
- Think: Provide thorough reasoning before taking any action. Include the rationale for the selected tool and expected outcome.
    - Format: <Think> ... </Think>
- Action: Choose which tag to take actions after. 
    - Format: <Action> Target IDs (or Document) | Action </Action>
        - Part 1: Target X, Y, Z (e.g., Target 1, 2, 3 for documents or queries; Target 0 for user input).
        - Part 2: Operation type (e.g., Query Search, Detail Search, Query Rewrite).
- End: Terminate the interaction immediately after this tag.
    - Format: <END>

**Example 1**
Action History: 
Nothing yet.
Query Collected: 
<Query 0> When was the first computer invented? 
Documents Retrieved: 
Nothing yet. 
User Input: 
When was the first computer invented? 

**Expected Output Example 1**: 
<Think> The input is straightforward, but direct Query 0 search may introduce noise with irrelevant terms. Rewriting the query first will improve retrieval precision for the exact invention date. </Think>
<Action> Target 0 | Query Rewrite </Action>
<Think> After Query Rewrite generated 5 refined queries, we initiate Query Search on Query 1, 2, 3 to find the exact computer invention date. </Think> 
<Action> Target 1, 2, 3 | Query Search </Action>
<Think> Query Search generated mixed relevance documents; Document Filter will eliminate noise. With all three action slots exhausted, we pause for state review before next action. </Think>
<Action> Target 0 | Document Filter </Action>
<END>

**Example 2**
Action History: 
[1. Action: Query Search | Target ID: 0] -> 1. [Action: Query Rewrite | Target ID: 0] -> 2. [Action: Query Search | Target ID: 2] -> 2. [Action: Document Filter | Target ID: 3, 4, 5] 
Query Collected: 
<Query 0> <Already Retrieved> How does artificial intelligence impact the healthcare industry?
<Query 1> AI healthcare impact
<Query 2> <Already Retrieved> Medical AI applications
<Query 3> AI in health services
<Query 4> Healthcare AI transformation
<Query 5> AI medical sector effects
Documents Retrieved: 
<Document 0> [Title] AI Healthcare Impact Analysis Prompt [Context] Analyze artificial intelligence's impact on healthcare, emphasizing diagnostic accuracy, personalized treatment, and operational efficiency. Include benefits like reduced costs and ethical challenges such as data privacy...
<Document 1> [Title] Urban Tech Integration Overview [Context] Explore the evolving role of digital systems in city environments, emphasizing connectivity and sustainability. Discuss general advancements in infrastructure and community engagement without specific case studies. Maintain a broad focus on societal adaptation to technological shifts in daily urban life.
<Document 2> [Title] Sustainable Gardening Practices Guide [Context] Outline eco-friendly gardening techniques, including composting, water conservation, and native plant selection. Focus on reducing chemical use and enhancing soil health. Highlight benefits like biodiversity support and lower environmental impact. Provide actionable tips for beginners to create thriving, low-maintenance gardens.
User Input: 
How does artificial intelligence impact the healthcare industry?

**Expected Output Example 2**: 
<Think> Document 0 directly addresses AI in healthcare with specific focus on diagnostics and ethics, but lacks depth on service applications. Action History shows Query 0 (original) and Query 2 (Medical AI applications) were already retrieved. To fill knowledge gaps in health services context, Query 3 ('AI in health services') needs to be searched next. This will provide complementary insights without redundancy. </Think>
<Action> Target 3 | Query Search </Action>
<Think> Document 1 (Urban Tech) and Document 2 (Gardening) are clearly irrelevant to healthcare AI queries. To systematically remove all irrelevant entries before final output, Document Filter is the optimal tool—it efficiently processes the entire list in one action, eliminating future redundancy without manual iteration. This aligns with the guideline to reject borderline-relevance documents immediately. </Think>
<Action> Target 1, 2 | Document Filter </Action>
<Think> After Document Filter action removed Documents 1 and 2, only Document 0 remains as highly relevant. With one action budget left and no further tool needed for verification, we avoid unnecessary actions. The next interaction will confirm Document 0's sufficiency before finalizing with Stop. </Think>
<END>

""" + \
f"""
Current State:

**Actions History:**
Nothing Yet. 

**Queries Collected:**
This is a test...

**Documents Retrieved:**
Nothing Yet

**User Input:**
**This is a test...**

Your Output: 
"""

inputs = tokenizer(
    [test_string],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048
)

input_ids = inputs["input_ids"].to(next(model.parameters()).device)
attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)



print(tokenizer.decode(input_ids[0]))
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens = 2000, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, 
    ) # 需要将结果移到内存中，也即CPU上，不能指向GPU内存，不够用


tokenizer.decode(outputs[0][len(input_ids[0]):])