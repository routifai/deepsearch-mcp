#!/usr/bin/env python3
"""
Search Orchestrator
Coordinates query analysis, search execution, and content fetching.
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from configurations.config import config
from agents.query_analyzer import QueryAnalyzer, QueryAnalysis, SearchStrategy, ComplexityLevel
from tools.search_engine import ModularSearchEngine, SearchCategory
from tools.web_fetcher import WebFetcher
from configurations.exceptions import handle_error, SearchServerError

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    """Coordinates the entire search process"""
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.search_engine = ModularSearchEngine()
        self.web_fetcher = WebFetcher()
        
    async def search_and_fetch(self, query: str) -> Dict[str, Any]:
        """
        Main entry point: analyze query, execute searches, and fetch content
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting search orchestration for: '{query}'")
            
            # Step 1: Analyze the query
            analysis = await self.analyzer.analyze_query(query)
            
            logger.info(f"Analysis complete: {analysis.search_strategy}, "
                       f"{len(analysis.search_queries)} searches planned")
            
            # Step 2: Execute search strategy
            if analysis.search_strategy == SearchStrategy.NO_SEARCH:
                return self._create_no_search_response(query, analysis, start_time)
            
            search_results = await self._execute_searches(analysis.search_queries)
            
            # Step 3: Fetch content from promising URLs
            fetch_results = await self._fetch_content(search_results, analysis)
            
            # Step 4: Format response
            response = self._create_response(
                query, analysis, search_results, fetch_results, start_time
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Search orchestration complete in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = handle_error(e, "search_orchestration")
            logger.error(f"Search orchestration failed after {processing_time:.2f}s: {e}")
            
            return {
                "query": query,
                "error": error_msg,
                "processing_time": processing_time,
                "content": f"Search failed: {error_msg}"
            }
    
    async def _execute_searches(self, search_queries: List) -> Dict[str, Any]:
        """Execute all planned search queries"""
        
        results = {}
        all_urls = []
        
        for i, search_query in enumerate(search_queries):
            try:
                # Convert category string to enum
                category = SearchCategory(search_query.category)
                
                # Execute search
                search_results = await self.search_engine.search(
                    query=search_query.query,
                    category=category,
                    num_results=8  # Get more results for better URL selection
                )
                
                # Collect URLs for fetching
                urls = [result.url for result in search_results]
                all_urls.extend(urls[:5])  # Take top 5 URLs per search
                
                results[f"search_{i+1}"] = {
                    "query": search_query.query,
                    "category": search_query.category,
                    "rationale": search_query.rationale,
                    "results": [
                        {
                            "title": r.title,
                            "url": r.url,
                            "snippet": r.snippet,
                            "rank": r.rank
                        }
                        for r in search_results
                    ],
                    "url_count": len(search_results)
                }
                
                logger.info(f"Search {i+1} completed: {len(search_results)} results")
                
            except Exception as e:
                logger.error(f"Search {i+1} failed: {e}")
                results[f"search_{i+1}"] = {
                    "query": search_query.query,
                    "category": search_query.category,
                    "rationale": search_query.rationale,
                    "error": str(e),
                    "url_count": 0
                }
        
        results["all_urls"] = list(dict.fromkeys(all_urls))  # Remove duplicates
        return results
    
    async def _fetch_content(self, search_results: Dict[str, Any], 
                           analysis: QueryAnalysis) -> Dict[str, str]:
        """Fetch content from promising URLs"""
        
        urls = search_results.get("all_urls", [])
        if not urls:
            logger.info("No URLs to fetch")
            return {}
        
        # Determine fetch parameters based on complexity
        if analysis.complexity_level == ComplexityLevel.SIMPLE:
            fetch_mode = "snippet"
            target_successful = 2
        elif analysis.complexity_level == ComplexityLevel.MODERATE:
            fetch_mode = "partial"
            target_successful = 3
        else:  # COMPLEX
            fetch_mode = "partial"
            target_successful = 5
        
        # Limit total URLs to try
        urls_to_fetch = urls[:15]  # Max 15 URLs to avoid overwhelming
        
        logger.info(f"Fetching content: {len(urls_to_fetch)} URLs, "
                   f"target {target_successful} successful, mode: {fetch_mode}")
        
        try:
            fetch_results = await self.web_fetcher.fetch_multiple_urls(
                urls=urls_to_fetch,
                mode=fetch_mode,
                target_successful=target_successful
            )
            
            # Count successful fetches
            successful = sum(1 for content in fetch_results.values() 
                           if not content.startswith("❌"))
            
            logger.info(f"Content fetching complete: {successful} successful")
            return fetch_results
            
        except Exception as e:
            logger.error(f"Content fetching failed: {e}")
            return {}
    
    def _create_no_search_response(self, query: str, analysis: QueryAnalysis, 
                                 start_time: datetime) -> Dict[str, Any]:
        """Create response when no search is needed"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "strategy": "no_search",
            "reasoning": analysis.reasoning,
            "processing_time": processing_time,
            "content": "No search needed - this can be answered from existing knowledge.",
            "analysis": {
                "temporal_category": analysis.temporal_category,
                "complexity_level": analysis.complexity_level,
                "requires_current_data": analysis.requires_current_data
            }
        }
    
    def _create_response(self, query: str, analysis: QueryAnalysis, 
                        search_results: Dict[str, Any], fetch_results: Dict[str, str],
                        start_time: datetime) -> Dict[str, Any]:
        """Create the final response"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract successful content for LLM
        successful_content = {
            url: content for url, content in fetch_results.items() 
            if not content.startswith("❌")
        }
        
        # Format content for LLM consumption
        if successful_content:
            content_parts = []
            for url, content in successful_content.items():
                # Clean the content to remove metadata
                clean_content = self._clean_content_for_llm(content)
                content_parts.append(f"**Source: {url}**\n{clean_content}")
            
            llm_content = "\n\n---\n\n".join(content_parts)
        else:
            llm_content = "No content could be extracted from web sources."
        
        return {
            "query": query,
            "strategy": analysis.search_strategy,
            "processing_time": processing_time,
            "content": llm_content,
            "analysis": {
                "temporal_category": analysis.temporal_category,
                "complexity_level": analysis.complexity_level,
                "requires_current_data": analysis.requires_current_data,
                "reasoning": analysis.reasoning
            },
            "stats": {
                "searches_executed": len([k for k in search_results.keys() if k.startswith("search_")]),
                "urls_found": len(search_results.get("all_urls", [])),
                "urls_fetched": len(fetch_results),
                "successful_fetches": len(successful_content)
            }
        }
    
    def _clean_content_for_llm(self, content: str) -> str:
        """Clean content for LLM consumption by removing metadata"""
        
        lines = content.split('\n')
        cleaned_lines = []
        skip_metadata = False
        
        for line in lines:
            # Skip metadata section
            if line.startswith("**Source:**") or line.startswith("**URL:**") or \
               line.startswith("**Extracted:**") or line.startswith("**Mode:**"):
                skip_metadata = True
                continue
            elif line.startswith("---"):
                skip_metadata = False
                continue
            elif skip_metadata:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        
        return {
            "search_engine": self.search_engine.get_status(),
            "query_analyzer": self.analyzer.get_cache_stats(),
            "config": {
                "max_concurrent": config.MAX_CONCURRENT,
                "timeout_seconds": config.TIMEOUT_SECONDS,
                "max_cache_size": config.MAX_CACHE_SIZE
            }
        }