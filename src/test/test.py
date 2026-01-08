from lib.test.test_train import *
from lib.test.test_reward import *
from lib.test.test_AERR import *
from lib.test.test_dataset import *
from lib.test.test_llms import *



from lib.test.test_AERR import DummyModel, DummyIndexer


# config = AERRConfig(test = True)
# config.decision.load_without_model = True

# pipeline = AERR(config = config)

# pipeline.decision_agent.model = DummyModel()
# pipeline.excution_agent.indexer = DummyIndexer()
# pipeline.excution_agent.model = DummyModel()

# output = pipeline.generate_batch(input_prompts = ['What is the capital of France?'])

# print(output)

from lib.large_language_model import Large_Language_Model_API
import re

llm_api = Large_Language_Model_API()
llm_api.init_llm("")

@ staticmethod
def Execution_document_filter_input_prompt(user_input: str, documents: List[str]) -> str:
    """
    根据用户查询和文档内容，判断文档是否与查询相关，并输出过滤结果。

    Args:
        documents (str): 文档内容（格式：[Title] ... [Context] ...）

    Returns:
        str: 完整的提示模板。
    """
    docs_str = "\n".join([f"<Document {i}>: {d[:200]}" for i, d in enumerate(documents)]) if len(documents) != 0 else "Nothing yet."
    prompt = f"""
Based on the user input and the provided documents, determine if each document is relevant to the query.

**Document Relevance Criteria**

**RELEVANT (Keep) - Document must meet ONE of these criteria:**
1. **Direct Answer**: Explicitly provides specific information that directly answers the user's question
   - Example: For "When was X invented?", document states "X was invented in 1945"
2. **Causal Explanation**: Explains how/why something happens in relation to the query
   - Example: For "How does climate change affect agriculture?", document explains specific impact mechanisms
3. **Quantitative Evidence**: Provides statistical data, measurements, or quantified results relevant to the query
4. **Specific Case Study**: Details a concrete example that illustrates the answer to the query

**IRRELEVANT (Discard) - Document falls under ONE of these categories:**
1. **Topic Mention Only**: Mentions query keywords but provides no substantive information answering the core question
2. **Background Context**: Provides only general background without addressing the specific query
3. **Future/Potential**: Discusses potential future developments without current evidence
4. **Completely Unrelated**: Primary subject has no meaningful connection to the query
5. **Methodology Focus**: Focuses on research methods without substantive findings relevant to the query

**Decision Guidelines**
- **Precision over Recall**: When in doubt, mark as irrelevant
- **Concrete over Vague**: Require specific, actionable information
- **Direct over Indirect**: Prefer documents that directly address the query vs. tangential connections

**Response Format Specification**
- For EACH document, provide:
  <Think Document i> [Required] 1 sentenice reasoning for this specific document
  <Judgment Document i> [Required] "Relevant" or "Irrelevant"
- After analyzing all documents, provide summary:
  <Relevant Documents> [Required] Comma-separated indices of all relevant documents
  <Irrelevant Documents> [Required] Comma-separated indices of all irrelevant documents

**Example 1**:
Documents: 
<Document 0> [Title] The Invention of the First Electronic Computer [Context] The first general-purpose electronic digital computer, ENIAC, was completed in 1945 at the University of Pennsylvania and publicly unveiled on February 15, 1946. It was designed by J. Presper Eckert and John Mauchly, using 17,468 vacuum tubes.
<Document 1> [Title] Early Mechanical Calculating Devices [Context] Charles Babbage's Analytical Engine (1837) and Herman Hollerith's tabulating machine (1890) were precursors to electronic computers, but these mechanical devices did not use electronic components for computation.
<Document 2> [Title] Evolution of Computer Hardware Components [Context] The transition from vacuum tubes to transistors in the 1950s marked a key shift in computer design, enabling smaller and more reliable machines, though this hardware evolution occurred decades after the first electronic computers.
<Document 3> [Title] Personal Computer Adoption in the 1980s [Context] The Apple II and IBM PC popularized personal computing in the 1980s, transforming computers from specialized research tools into consumer devices, but this development occurred long after the initial invention.
<Document 4> [Title] The Development of the Steam Engine [Context] Thomas Newcomen invented the steam engine in 1712, with James Watt's improvements in the 1760s powering the Industrial Revolution, unrelated to computing technology.
<Document 5> [Title] World War II's Role in Computer Development [Context] Military demands for rapid computation during World War II accelerated electronic computer research, leading to projects like ENIAC, though the exact timeline of the first completed machine remains historically debated.

User's Input: When was the computer invented? 
Expected Output:
<Think Document 0> Directly states ENIAC's completion in 1945 and unveiling in 1946, providing specific invention timeline.
<Judgment Document 0> Relevant
<Think Document 1> Discusses mechanical precursors but lacks electronic computer invention date.
<Judgment Document 1> Irrelevant
<Think Document 2> Covers hardware evolution without initial invention information.
<Judgment Document 2> Irrelevant
<Think Document 3> Focuses on later adoption, not invention.
<Judgment Document 3> Irrelevant
<Think Document 4> Completely unrelated to computing.
<Judgment Document 4> Irrelevant
<Think Document 5> Mentions acceleration of research but does not specify invention date.
<Judgment Document 5> Irrelevant
<Relevant Documents> 0
<Irrelevant Documents> 1,2,3,4,5

**Example 2**:
Documents: 
<Document 0> [Title] Water Scarcity in Farming [Context] A comprehensive assessment of water scarcity in the Murray-Darling Basin, Australia (2010-2023), showing that climate change has reduced average annual streamflow by 20% and increased drought frequency from once every 10 years to once every 3 years. This has led to a 30% reduction in irrigated area and a 25% decline in agricultural output in the basin, with detailed hydrological modeling and farmer survey results from 200 farms across the region.
<Document 1> [Title] Impact of Temperature Rise on Soil Fertility [Context] Laboratory and field experiments conducted at the University of California, Davis, from 2020-2023 showing that a 2°C increase in soil temperature reduces nitrogen mineralization rates by 15-20% and increases organic matter decomposition by 25% in temperate soils. The study includes data from 15 different soil types across California, with implications for fertilizer requirements and long-term soil health management, including case studies from vineyards and almond orchards.
<Document 2> [Title] Agricultural Adaptation Strategies [Context] A comparative analysis of drought-tolerant maize varieties (e.g., DTMA-12, DTMA-15) versus conventional varieties in the US Great Plains (2019-2023), showing a 25% yield increase during drought years and a 12% yield increase during normal years. Includes data on precision irrigation systems' water savings (up to 35% water reduction with 8% yield increase) and cost-effectiveness analysis for smallholder farmers, with case studies from 50 farms in Nebraska and Kansas.
<Document 3> [Title] Climate Change and Crop Production [Context] Analysis of IPCC AR6 data (2023) showing that for every 1°C increase in global temperature, wheat yields decrease by 6.0% (95% CI: 4.5-7.5%), rice by 3.2% (95% CI: 2.0-4.4%), and maize by 7.4% (95% CI: 5.8-9.0%) in major producing regions. The study includes regional case studies from India (wheat), Thailand (rice), and the US Midwest (maize), with data from 2000-2022, including yield maps and statistical analysis.
<Document 4> [Title] Urban Agriculture Trends [Context] A survey of 200 urban farming initiatives in 10 major cities (New York, Tokyo, Nairobi, etc.) conducted in 2022-2023, showing that urban agriculture contributes an average of 5% of fresh vegetable supply in these cities, with a 15% growth rate annually. The study focuses on food security benefits in urban areas, community engagement, and economic viability, with minimal discussion of climate change impacts on traditional agriculture.
<Document 5> [Title] Sustainable Farming Methods [Context] A 5-year field study (2018-2023) conducted across 120 farms in Brazil, India, and Kenya comparing conventional farming with agroforestry systems. Results show agroforestry plots had 22% higher soil moisture retention during droughts, 18% lower pest incidence, and 15% higher yields in climate-vulnerable regions compared to conventional farms. Includes detailed implementation protocols and cost-benefit analysis for smallholder farmers.
<Document 6> [Title] Global Food Security Challenges [Context] Analysis of FAO and World Bank data (2020-2023) showing that climate-related yield reductions have contributed to a 15% increase in global food prices since 2019, with the most severe impacts in Sub-Saharan Africa and South Asia. The study projects that without adaptation, climate change could push an additional 80 million people into food insecurity by 2050, with detailed regional breakdowns and case studies from Ethiopia, Pakistan, and Bangladesh.
<Document 7> [Title] Renewable Energy in Agriculture [Context] Implementation report on solar-powered irrigation systems in 500 farms across Rajasthan, India (2021-2023), showing a 40% reduction in diesel consumption for irrigation, with an average payback period of 3.2 years. The study focuses on energy efficiency and cost savings, with minimal discussion of climate change impacts on agriculture, including detailed cost analysis and farmer testimonials.
<Document 8> [Title] Climate-Driven Pest Outbreaks [Context] A case study from 2022-2023 showing how the expansion of the fall armyworm (Spodoptera frugiperda) in East Africa due to warmer temperatures has resulted in a 35% average yield loss for maize farmers, with total economic losses estimated at $1.2 billion in Kenya, Tanzania, and Uganda. The study includes detailed maps of pest spread, temperature correlation data, and farmer survey results from 300 farms across the region.
<Document 9> [Title] Economic Effects of Climate Change [Context] A World Bank study (2022) analyzing the economic impacts of climate change across 100 countries, with agriculture representing 30% of the total economic impact. The study shows that agricultural losses account for 45% of the total climate-related economic damage, with the most significant impacts in low-income countries. Includes macroeconomic modeling of GDP impacts and trade effects, with limited specific agricultural case studies.

User's Input: How does climate change affect agriculture?
Expected Output:
<Think Document 0> Directly shows climate change causing water scarcity that reduces agricultural output by 25%.
<Judgment Document 0> Relevant
<Think Document 1> Explains temperature effects on soil processes affecting agriculture.
<Judgment Document 1> Relevant
<Think Document 2> Focuses on adaptation solutions rather than climate impacts.
<Judgment Document 2> Irrelevant
<Think Document 3> Provides quantitative evidence of temperature impacts on crop yields.
<Judgment Document 3> Relevant
<Think Document 4> Urban agriculture trends with minimal climate impact discussion.
<Judgment Document 4> Irrelevant
<Think Document 5> Sustainable methods as solutions, not direct climate impacts.
<Judgment Document 5> Irrelevant
<Think Document 6> Links climate-related yield reductions to food security impacts.
<Judgment Document 6> Relevant
<Think Document 7> Renewable energy applications, not climate impacts on agriculture.
<Judgment Document 7> Irrelevant
<Think Document 8> Shows climate-driven pest outbreaks causing yield losses.
<Judgment Document 8> Relevant
<Think Document 9> Quantifies economic impacts of climate change on agriculture.
<Judgment Document 9> Relevant
<Relevant Documents> 0,1,3,6,8,9
<Irrelevant Documents> 2,4,5,7

**Your Task:**
**Documents:**
{docs_str}

**User's Input:**
{user_input}

**Your Output:**
""".strip()
    return prompt


User_Query = "How does plastic pollution impact coral reef ecosystems?"
documents = [
"[Title] Coral Plastic Smothering Study [Context] Plastic debris covering coral colonies reduced growth rates by 32% in Great Barrier Reef monitoring sites."
,"[Title] Microplastics and Coral Bleaching [Context] Microplastics in water column increased coral bleaching events by 45% during 2020-2023."
,"[Title] Coral Reef Restoration Project [Context] After removing 15 tons of plastic from reef sites, coral recovery rates increased by 28% within 18 months."
,"[Title] Plastic Toxicity Report [Context] Plastic leachates caused 60% higher mortality in coral polyps compared to control groups."
,"[Title] Marine Debris Policy [Context] 2022 UN resolution targeting plastic reduction in coral zones led to 22% decrease in plastic accumulation in Pacific reefs."
,"[Title] Ocean Plastic Recycling Rates [Context] Global plastic recycling rate reached 9% in 2023. Discusses collection infrastructure, not marine impacts."
,"[Title] Whale Migration Patterns [Context] Satellite tracking of humpback whales in Atlantic Ocean. No plastic pollution data."
,"[Title] Ocean Acidification Study [Context] pH levels dropped 0.1 units in 2023, affecting shellfish. No plastic mentions."
,"[Title] Sea Turtle Nesting Reports [Context] 2022 nesting success rates at 65% on Pacific islands. Mentions plastic ingestion but not coral impacts."
,"[Title] Coastal Tourism Survey [Context] Visitor numbers increased 15% in 2023. No environmental impact metrics."

]

prompts = Execution_document_filter_input_prompt(user_input = User_Query, documents = documents)
result = llm_api.generate(prompts)
print(result)

@staticmethod
def Execution_parse_irrelevant_documents(model_output: str) -> List[str]:
    """
    从模型输出中提取不相关的文档索引，支持新旧两种格式
    """
    # 支持两种标签格式：单数和复数
    patterns = [
        r'<Irrelevant Documents>\s*([\d, ]+)',  # 新格式：复数
        r'<Irrelevant Document>\s*([\d, ]+)'    # 旧格式：单数
    ]
    
    all_indices = []
    for pattern in patterns:
        matches = re.findall(pattern, model_output)
        for match in matches:
            indices = [idx.strip() for idx in match.split(',')]
            for idx in indices:
                all_indices.append(int(idx))
    
    # 去重并排序
    all_indices = list(set(all_indices))
    all_indices.sort()
    return [str(idx) for idx in all_indices]


def DocumentFilter(document_list: List[str], response: str) -> None:
    """
    更安全的文档过滤方法
    """
    if isinstance(response, list):
        response = response[0]
    
    # 解析不相关文档索引
    irrelevant_indices = AERRTemplate.Execution_parse_irrelevant_documents(response)
    
    if not irrelevant_indices:
        return  # 没有不相关文档，直接返回
    
    try:
        irrelevant_indices_int = [int(idx) for idx in irrelevant_indices]
    except ValueError:
        return  # 索引格式错误
    
    # 使用集合提高查找效率，并去重
    irrelevant_set = set(irrelevant_indices_int)
    
    # 使用列表推导式创建新列表（更安全的方法）
    filtered_documents = [
        doc for i, doc in enumerate(document_list) 
        if i not in irrelevant_set
    ]
    
    # 清空原列表并添加过滤后的文档
    document_list.clear()
    document_list.extend(filtered_documents)

print(Execution_parse_irrelevant_documents(result[0]))
DocumentFilter(document_list = documents, response = result)
print(documents)
