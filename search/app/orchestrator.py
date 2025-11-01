# Main orchestrator for the search app
# orchestrator.py - Main orchestrator

# Steps:
# 0. Parse/preprocess query
# 1. Decipher query intent
# 2. Decide retriever/services/tools strategy
# 3. Execute searches in parallel
# 4. Combine and rank results
# 5. Filter results
# 6. Return results

import time
from typing import List, Optional, Dict, Any

from retrievers.specifications import create_retriever_dict

from utils.logging import setup_logger

logger = setup_logger(__name__)

class SearchOrchestrator:
    def __init__(self):
        # Initialize retrievers (lazy loading could be used here)
        # TODO: use a registry instead of hardcoding the retrievers
        # TODO: use pydantic to nake all these pareamter and attriobitues better defined and validated
        self.retrievers = create_retriever_dict()
        self.allowed_search_types = ["track", "album", "artist", "playlist", "combined"]
    
    def search(
        self, 
        query: str, 
        search_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main search orchestration method.
        
        Args:
            query: Search query string
            search_types: Optional list of search types to use. 
                         If None, will auto-select based on query.
            top_k: Number of results per search type
        
        Returns:
            Combined and ranked search results
        """

        # Step 0 : Parse/preprocess query
        #preprocessed_query = self._preprocess_query(query)

        # Step 1 : Decipher query intent
        #query_intent, search_types, tools_to_use = self._decipher_query_intent(preprocessed_query)

        # TODO: continue

        # Step 2 : Decide retriever/services/tools strategy
        # If search_types not specified, decide which retrievers to use
        if search_types is None:
        #    search_types = self._decide_search_types(query)
             search_types = self.allowed_search_types

        # Step 3 : Execute searches in parallel
        search_results = self._retrieve_results(query, search_types, top_k)
        
        # Step 4 : Rank results
        #ranked_results = self._rank_results(search_results, top_k)

        # Step 5 : Filter results
        #filtered_results = self._filter_results(ranked_results, top_k)
        
        return {
            "raw_results": search_results,
            #"ranked_results": ranked_results,
            "search_types": search_types
        }

    def _retrieve_results(self, query: str, search_types: List[str], top_k: int) -> Dict[str, List[Dict]]:
        """
        Retrieve results from the retrievers.
        """
        total_start_time = time.time()
        results = {}
        for search_type in search_types:
            if search_type in self.retrievers:
                try:
                    start_time = time.time()
                    logger.info(f"Retrieving results for {search_type} with query: {query}")
                    results[search_type] = self.retrievers[search_type].search(
                        query=query, 
                        top_k=top_k
                    )
                    elapsed_time = time.time() - start_time
                    logger.info(f"Retrieved results for {search_type} in {elapsed_time:.3f} seconds")
                except Exception as e:
                    # Log error but continue with other retrievers
                    logger.error(f"Error in {search_type} search: {e}")
        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total search time: {total_elapsed_time:.3f} seconds")
        return results

    def _decide_search_types(self, query: str) -> List[str]:
        """
        Decide which retrievers to use based on query.
        Could be enhanced with query classification.
        """
        query_lower = query.lower()
        
        # Simple heuristics (could be replaced with ML classifier)
        if any(keyword in query_lower for keyword in ["album", "album:"]):
            return ["album", "combined"]
        elif any(keyword in query_lower for keyword in ["artist", "by"]):
            return ["artist", "combined"]
        elif any(keyword in query_lower for keyword in ["playlist"]):
            return ["playlist", "combined"]
        else:
            # Default: try track and combined
            return ["track", "combined"]
    
    def _combine_results(
        self, 
        results: Dict[str, List[Dict]], 
        top_k: int
    ) -> List[Dict]:
        """
        Combine results from multiple retrievers.
        Could implement more sophisticated ranking/fusion here.
        """
        all_results = []
        seen_tracks = set()
        
        for search_type, search_results in results.items():
            for result in search_results:
                # Create unique key to deduplicate
                track_key = (
                    result.get("track_name", ""),
                    result.get("artist_name", "")
                )
                
                if track_key not in seen_tracks:
                    result["search_type"] = search_type
                    all_results.append(result)
                    seen_tracks.add(track_key)
        
        # Sort by score and return top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]