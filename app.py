# File: portfolio_manager.py
import os
import json
import yfinance as yf
import pandas as pd
from typing import TypedDict
from fredapi import Fred
from sec_edgar_downloader import Downloader
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI
from neo4j import GraphDatabase
from pyportfolioopt import EfficientFrontier, expected_returns, risk_models

# ----------------- Configuration -----------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")

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
llm = AzureChatOpenAI(
    deployment_name="gpt-4-portfolio",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint="https://your-resource.openai.azure.com/"
)

# ----------------- Enhanced Data Ingestion -----------------
class DataIngestor:
    @staticmethod
    def get_market_data(period: str = "3y"):
        """Fetch historical prices for all portfolio assets"""
        return yf.download(PORTFOLIO_SYMBOLS, period=period)['Adj Close']

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
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def create_portfolio_structure(self):
        """Initialize portfolio nodes and relationships"""
        with self.driver.session() as session:
            # Create asset classes
            session.execute_write(lambda tx: tx.run(
                "MERGE (:AssetClass {name: 'Equities'})"
                "MERGE (:AssetClass {name: 'Fixed Income'})"
                "MERGE (:AssetClass {name: 'Commodities'})"
            ))
            
            # Link assets to classes
            for symbol in PORTFOLIO_SYMBOLS:
                asset_class = 'Commodities' if symbol == GOLD_SYMBOL else \
                            'Fixed Income' if symbol in BOND_SYMBOLS else \
                            'Equities'
                session.execute_write(lambda tx: tx.run(
                    f"MERGE (a:Asset {{symbol: '{symbol}'})"
                    f"MERGE (c:AssetClass {{name: '{asset_class}'}})"
                    "MERGE (a)-[:BELONGS_TO]->(c)"
                ))

# ----------------- Enhanced LangGraph Workflow -----------------
class PortfolioState(TypedDict):
    user_query: str
    symbols: list
    economic_data: dict
    market_data: pd.DataFrame
    risk_factors: dict
    constraints: dict
    weights: dict

def extract_requirements(state: PortfolioState):
    prompt = """Analyze portfolio request and extract:
    {query}

    Return JSON with:
    - risk_tolerance: low/medium/high
    - time_horizon: years
    - constraints: {
        max_sector_allocation: {sector: max_percent},
        min_alternative_allocation: percentage
    }"""
    response = llm.invoke(prompt.format(query=state["user_query"]))
    return json.loads(response.content)

def analyze_risks(state: PortfolioState):
    prompt = """Analyze market risks for a portfolio containing {symbols} given:
    Economic Indicators: {econ_data}
    1-Year Volatility: {volatility}
    
    Output risk mitigation strategy as JSON with:
    - required_bond_allocation: percentage
    - required_gold_allocation: percentage
    - sector_risk_adjustments: {sector: adjusted_max_allocation}"""
    
    econ_data = {k: v.iloc[-1] for k, v in state["economic_data"].items()}
    volatility = state["market_data"].pct_change().std().mean()
    
    response = llm.invoke(prompt.format(
        symbols=PORTFOLIO_SYMBOLS,
        econ_data=econ_data,
        volatility=round(volatility, 4)
    ))
    return {"risk_factors": json.loads(response.content)}

def optimize_portfolio(state: PortfolioState):
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
    for sector, max_alloc in {**state["constraints"]["max_sector_allocation"],
                              **state["risk_factors"]["sector_risk_adjustments"]}.items():
        sector_indices = [i for i, s in enumerate(sector_map) if s == sector]
        if sector_indices:
            ef.add_constraint(lambda w, si=sector_indices: sum(w[si]) <= max_alloc)
    
    # Optimize based on risk tolerance
    if state["constraints"]["risk_tolerance"] == 'low':
        ef.min_volatility()
    elif state["constraints"]["risk_tolerance"] == 'medium':
        ef.max_sharpe()
    else:
        target_return = returns.mean() * 1.2  # 20% higher than average
        ef.efficient_return(target_return)
    
    return {"weights": ef.clean_weights()}

def generate_report(state: PortfolioState):
    prompt = """Generate comprehensive portfolio analysis report with:
    - Current market risk assessment
    - Asset allocation rationale
    - Stress test scenarios
    - Rebalancing recommendations
    
    Portfolio Details:
    {weights}
    
    Risk Factors:
    {risk_factors}
    
    Economic Context:
    {econ_data}"""
    
    econ_data = "\n".join([f"{k}: {v.iloc[-1]:.2f}" 
                         for k, v in state["economic_data"].items()])
    
    response = llm.invoke(prompt.format(
        weights=json.dumps(state["weights"], indent=2),
        risk_factors=json.dumps(state["risk_factors"], indent=2),
        econ_data=econ_data
    ))
    return {"report": response.content}

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
    
    state = {
        "user_query": query,
        "symbols": PORTFOLIO_SYMBOLS
    }
    
    # Extract and merge requirements
    state.update(extract_requirements(state))
    
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
    print(result["report"])