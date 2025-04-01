# File: portfolio_manager.py
import os
import json
import yfinance as yf
import pandas as pd
from typing import TypedDict
from fredapi import Fred
from sec_edgar_downloader import Downloader
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser()
strparser = StrOutputParser()


# ----------------- Configuration -----------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")


from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Predefined portfolio constituents
PORTFOLIO_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',
    'JNJ', 'PG', 'V', 'MA', 'NVDA',          # 10 stocks
    'GLD',                                   # Gold ETF
    'TLT', 'BND'                             # Bond ETFs
]

SYMBOL_SECTORS = {
    # Stocks
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOG': 'Technology',
    'AMZN': 'Consumer Discretionary',
    'TSLA': 'Consumer Discretionary',
    'JNJ': 'Healthcare',
    'PG': 'Consumer Staples',
    'V': 'Financials',
    'MA': 'Financials',
    'NVDA': 'Technology',
    
    # Alternative assets
    'GLD': 'Commodity',
    'TLT': 'Bonds',
    'BND': 'Bonds'
}

GOLD_SYMBOL = 'GLD'
BOND_SYMBOLS = ['TLT', 'BND']

# Initialize services
fred = Fred(api_key=FRED_API_KEY)

llm = ChatGroq(
    model="deepseek-r1-distill-qwen-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# ----------------- Enhanced Data Ingestion -----------------
class DataIngestor:
    @staticmethod
    def get_market_data(period: str = "3y"):
        """Fetch historical prices for all portfolio assets"""
        return yf.download(PORTFOLIO_SYMBOLS, period=period,auto_adjust=False)['Adj Close']

    @staticmethod
    def get_economic_data():
        """Fetch key economic indicators from FRED"""
        return {
            'DGS10': fred.get_series('DGS10'),   # 10-Year Treasury Rate
            'CPI': fred.get_series('CPIAUCSL'),  # Consumer Price Index
            'UNRATE': fred.get_series('UNRATE'), # Unemployment Rate
            'GFDEBTN': fred.get_series('GFDEBTN') # Federal Debt
        }

# ----------------- Enhanced Knowledge Graph -----------------
class FinancialKG:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j",  # Default database
            # aura_instance_id=os.getenv("AURA_INSTANCEID"),
            # aura_instance_name=os.getenv("AURA_INSTANCENAME")
        )
        
        # Verify connection
        try:
            self.graph.query("RETURN 1 AS test")
            print("Connected to Neo4j AuraDB successfully")
        except Exception as e:
            print(f"Connection failed: {str(e)}")

    def create_asset_node(self, symbol: str):
        self.graph.query(
            """MERGE (a:Asset {symbol: $symbol})
            RETURN a""",
            params={"symbol": symbol}
        )

    def link_risk_factors(self, symbol: str, risk_data: dict):
        self.graph.query(
            """MATCH (a:Asset {symbol: $symbol})
            MERGE (r:Risk {name: $risk_name})
            MERGE (a)-[rel:EXPOSED_TO]->(r)
            SET rel.score = $score""",
            params={
                "symbol": symbol,
                "risk_name": risk_data['name'],
                "score": risk_data['score']
            }
        )

    def create_portfolio_structure(self):
        """Initialize portfolio nodes and relationships"""
        # Create asset classes
        self.graph.query("""
            MERGE (:AssetClass {name: 'Equities'})
            MERGE (:AssetClass {name: 'Fixed Income'})
            MERGE (:AssetClass {name: 'Commodities'})
        """)
        
        # Link assets to classes
        for symbol in PORTFOLIO_SYMBOLS:
            asset_class = 'Commodities' if symbol == GOLD_SYMBOL else \
                        'Fixed Income' if symbol in BOND_SYMBOLS else \
                        'Equities'
            self.graph.query(
                """
                MERGE (a:Asset {symbol: $symbol})
                MERGE (c:AssetClass {name: $class})
                MERGE (a)-[:BELONGS_TO]->(c)
                """,
                params={"symbol": symbol, "class": asset_class}
            )
# ----------------- Enhanced LangGraph Workflow -----------------
class PortfolioState(TypedDict):
    user_query: str
    symbols: list
    economic_data: dict
    market_data: pd.DataFrame
    risk_factors: dict
    constraints: dict
    weights: dict
    requirements: dict

def extract_requirements(state: PortfolioState):
    # Safely get query from state
    user_query = state.get("user_query", "")
    
    template = """Analyze portfolio request and extract:
    {query}

    Return JSON with:
    - risk_tolerance: low/medium/high
    - time_horizon: years
    - constraints: {{
        max_sector_allocation: {{sector: max_percent}},
        min_alternative_allocation: percentage
    }}"""

    prompt = PromptTemplate(template=template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})
    

    chain = prompt | llm | parser
    response = chain.invoke({"query": user_query})

    print("output from extract_requirements:", response)
    
    # response = llm.invoke(prompt)
    return response


def analyze_risks(state: PortfolioState):
    template_risk = """Analyze market risks for a portfolio containing {symbols} given:
    Economic Indicators: {econ_data}
    1-Year Volatility: {volatility}
    
    Output JSON with:
    1. required_bond_allocation (0-1)
    2. required_gold_allocation (0-1) 
    3. sector_risk_adjustments (sector: max_allocation)
    4. risk_scores (1-5 scale)
    5. scenario_analysis (recession/rate_hike cases)

    format should be like this-

    dict : {{
            "required_bond_allocation": float (0-1),
            "required_gold_allocation": float (0-1),
            "sector_risk_adjustments": {{
                "Technology": float (0-1),
                "Healthcare": float (0-1),
                ...
            }},
            "risk_scores": {{
                "interest_rate_risk": int (1-5),
                "inflation_risk": int (1-5),
                "geopolitical_risk": int (1-5),
                "market_volatility": int (1-5)
            }},
            "scenario_analysis": {{
                "recession": {{
                    "expected_loss": float (0-1),
                    "recommended_actions": list[str]
                }},
                "rate_hike": {{
                    "expected_loss": float (0-1),
                    "recommended_actions": list[str]
                }}
            }}
        }}
    
    
    
    """


    prompt = PromptTemplate(
    template=template_risk,
    input_variables=["econ_data", "econ_data", "volatility"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    econ_data = {k: v.iloc[-1] for k, v in state["economic_data"].items()}
    volatility = state["market_data"].pct_change().std().mean()

    chain = prompt | llm | parser
    
    response = chain.invoke({"symbols":PORTFOLIO_SYMBOLS,"econ_data": econ_data, "volatility": volatility})
    
    # response = llm.invoke(prompt.format(
    #     symbols=PORTFOLIO_SYMBOLS,
    #     econ_data=econ_data,
    #     volatility=round(volatility, 4)
    # ))

    print("output from analyze_risks:\n", response)
    return {"risk_factors": response}

def optimize_portfolio(state: PortfolioState):
    print("optimize portfolio function called",state["requirements"])
    prices = state["market_data"]
    returns = expected_returns.mean_historical_return(prices)
    cov_matrix = risk_models.exp_cov(prices)
    
    ef = EfficientFrontier(returns, cov_matrix)
    
    # Basic diversification constraint
    ef.add_constraint(lambda w: w <= 0.15)  # Max 15% per stock
    
    # Add bond allocation constraint
    bond_indices = [i for i, s in enumerate(PORTFOLIO_SYMBOLS) if s in BOND_SYMBOLS]
    if bond_indices:
        ef.add_constraint(lambda w: sum(w[i] for i in bond_indices) >= 
                         state["risk_factors"]["required_bond_allocation"])
    
    # Add gold allocation constraint
    if GOLD_SYMBOL in PORTFOLIO_SYMBOLS:
        gold_index = PORTFOLIO_SYMBOLS.index(GOLD_SYMBOL)
        ef.add_constraint(lambda w: w[gold_index] >= 
                         state["risk_factors"]["required_gold_allocation"])
    
    # Sector constraints from both user and risk analysis
    sector_map = [SYMBOL_SECTORS[s] for s in PORTFOLIO_SYMBOLS]
    for sector, max_alloc in {**state["requirements"]["constraints"]["max_sector_allocation"],
                              **state["risk_factors"]["sector_risk_adjustments"]}.items():
        sector_indices = [i for i, s in enumerate(sector_map) if s == sector]
        if sector_indices:
            ef.add_constraint(lambda w, si=sector_indices: sum(w[si]) <= max_alloc)
    
    # Optimize based on risk tolerance
    if state["requirements"]["risk_tolerance"] == 'low':
        ef.min_volatility()
    elif state["requirements"]["risk_tolerance"] == 'medium':
        ef.max_sharpe()
    else:
        target_return = returns.mean() * 1.2  # 20% higher than average
        ef.efficient_return(target_return)
    
    return {"weights": ef.clean_weights()}

def generate_report(state: PortfolioState):
    report_prompt = """Generate comprehensive portfolio analysis report with:
    - Current market risk assessment
    - Asset allocation rationale
    - Stress test scenarios
    - Rebalancing recommendations
    
    Portfolio Details:
    {weights}
    
    Risk Factors:
    {risk_factors}
    
    Economic Context:
    {econ_data}
    
    ouptput should be in JSON format like this-

    {{"report": str}}
    
    """

    prompt = PromptTemplate(
    template=report_prompt,
    input_variables=["weights", "risk_factors", "econ_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    econ_data = "\n".join([f"{k}: {v.iloc[-1]:.2f}" 
                         for k, v in state["economic_data"].items()])
    
    # response = llm.invoke(prompt.format(
    #     weights=json.dumps(state["weights"], indent=2),
    #     risk_factors=json.dumps(state["risk_factors"], indent=2),
    #     econ_data=econ_data
    # ))

    response = chain.invoke({"weights":json.dumps(state["weights"], indent=2),"risk_factors": json.dumps(state["risk_factors"], indent=2), "econ_data": econ_data})

    print("output from generate_report:\n", response['report'])

    return {"generate_report":response}

# ----------------- Workflow Setup -----------------
workflow = StateGraph(PortfolioState)

workflow.add_node("ingest_data", lambda s: {
    "market_data": DataIngestor.get_market_data(),
    "economic_data": DataIngestor.get_economic_data()
})

workflow.add_node("analyze_risks", analyze_risks)
workflow.add_node("optimize", optimize_portfolio)
workflow.add_node("generate_report", generate_report)

workflow.set_entry_point("ingest_data")
workflow.add_edge("ingest_data", "analyze_risks")
workflow.add_edge("analyze_risks", "optimize")
workflow.add_edge("optimize", "generate_report")
workflow.add_edge("generate_report", END)

# ----------------- Execution -----------------
def run_portfolio_analysis(query: str):
    kg = FinancialKG()
    kg.create_portfolio_structure()
    
    # Initialize state with all required fields
    state = {
        "user_query": query,
        "symbols": PORTFOLIO_SYMBOLS,
        "economic_data": {},
        "market_data": pd.DataFrame(),
        "risk_factors": {},
        "constraints": {},
        "weights": {},
        "requirements": {}
    }
    
    # Add this verification step
    if "user_query" not in state:
        raise ValueError("State missing required 'user_query' field")
        
    # Extract and merge requirements
    requirements = extract_requirements(state)
    state.update({"requirements":requirements})
    
    # Add data ingestion to workflow
    # state.update({
    #     "market_data": DataIngestor.get_market_data(),
    #     "economic_data": DataIngestor.get_economic_data()
    # })
    
    # Run workflow
    app = workflow.compile()
    results = app.invoke(state)
    
    return results

# Example usage
if __name__ == "__main__":
    query = """Optimize portfolio for medium risk tolerance with:
    - Maximum 25% tech sector exposure
    - Minimum 20% allocation to safe-haven assets
    - 5-year investment horizon"""
    
    result = run_portfolio_analysis(query)
    print("\nPortfolio Recommendation:")
    print(result)