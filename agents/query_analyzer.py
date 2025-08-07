#!/usr/bin/env python3
"""
Simplified Query Analysis Engine
Removes complexity while maintaining intelligent query analysis.
"""

import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum
import logging
from datetime import datetime
import openai

from configurations.config import config
from configurations.cache import AnalysisCache
from configurations.exceptions import AnalysisError, LLMError

logger = logging.getLogger(__name__)

class TemporalCategory(str, Enum):
    CURRENT = "current"           # needs live/recent data
    RECENT = "recent"             # last few months
    HISTORICAL = "historical"     # stable over time
    TIMELESS = "timeless"         # concepts, definitions

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"             # single fact lookup
    MODERATE = "moderate"         # 2-3 sources needed
    COMPLEX = "complex"           # multi-faceted research

class SearchStrategy(str, Enum):
    NO_SEARCH = "no_search"       # answer from knowledge
    SINGLE_SEARCH = "single_search"   # one query sufficient
    MULTI_SEARCH = "multi_search"     # multiple searches needed

class SearchQuery(BaseModel):
    query: str = Field(description="The search query to execute")
    category: str = Field(description="Search category (general, news, academic, technical)")
    rationale: str = Field(description="Why this query is needed")

class QueryAnalysis(BaseModel):
    temporal_category: TemporalCategory
    complexity_level: ComplexityLevel
    search_strategy: SearchStrategy
    search_queries: List[SearchQuery] = Field(default_factory=list)
    requires_current_data: bool
    reasoning: str

class QueryAnalyzer:
    """Simplified query analyzer with caching"""
    
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise AnalysisError("OpenAI API key not configured")
        
        # Use centralized LLM client
        from agents.llm_client import get_llm_client
        llm_client = get_llm_client()
        self.client = llm_client.get_sync_client()
        self.instructor_client = instructor.from_openai(self.client)
        self.cache = AnalysisCache(
            maxsize=config.MAX_CACHE_SIZE,
            ttl_seconds=config.CACHE_TTL
        )
        self.knowledge_cutoff = config.get_knowledge_cutoff()
        
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query with caching"""
        return await self.cache.get_or_compute(
            query, 
            self._analyze_query_impl
        )
    
    async def _analyze_query_impl(self, query: str, context: dict = None) -> QueryAnalysis:
        """Implementation of query analysis"""
        
        system_prompt = f"""You are a query analysis expert. Determine how to best answer user queries.

Current date: {datetime.now().strftime('%Y-%m-%d')}
Knowledge cutoff: {self.knowledge_cutoff.strftime('%Y-%m-%d')}

TEMPORAL ANALYSIS:
- current: Information that changes frequently (news, current leaders, prices)
- recent: Information from last few months (recent events, developments)
- historical: Stable facts from before knowledge cutoff
- timeless: Concepts, definitions, established knowledge

COMPLEXITY LEVELS:
- simple: Single fact or straightforward question
- moderate: Requires 2-3 sources or basic comparison
- complex: Multi-faceted analysis or comprehensive research

SEARCH STRATEGY:
- no_search: Can be answered from existing knowledge
- single_search: One search query will suffice
- multi_search: Requires multiple targeted searches

For search queries, use these categories:
- general: Most factual queries
- news: Breaking news or current events
- academic: Research papers, studies
- technical: Documentation, programming help"""

        user_prompt = f"""Analyze this query: "{query}"

Consider:
1. Does this ask for information that could have changed since {self.knowledge_cutoff.strftime('%Y-%m-%d')}?
2. How complex is this query?
3. What search strategy would work best?

If searches are needed, create specific, targeted queries."""

        try:
            analysis = self.instructor_client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster, cheaper model for analysis
                response_model=QueryAnalysis,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            logger.info(f"Query analyzed: {analysis.search_strategy}, {len(analysis.search_queries)} queries")
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise LLMError(f"Failed to analyze query: {str(e)}")
    
    def enhance_query(self, original_query: str, analysis: QueryAnalysis) -> str:
        """Simple query enhancement without LLM calls"""
        
        # Add year for current queries
        if analysis.temporal_category == TemporalCategory.CURRENT:
            if "current" in original_query.lower() or "latest" in original_query.lower():
                current_year = datetime.now().year
                return f"{original_query} {current_year}"
        
        # Return original for everything else
        return original_query
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "hit_rate": self.cache.get_hit_rate(),
            "cache_stats": self.cache.cache.stats()
        }