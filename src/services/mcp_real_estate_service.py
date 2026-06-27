# # =====================================================================
# # src/services/mcp_real_estate_service.py (Imports section verification)
# # =====================================================================

# import pandas as pd
# from src.data.data_store import master_df

# # Clean connection to your standalone comparison engine bridge
# from src.services.comparison_service import run_comparison

# # Legacy agent imports
# from src.agents.analysis_agent import run_analysis
# from src.agents.negotiation_agent import run_negotiation_agent
# from src.agents.risk_agent import run_risk_agent
# from src.agents.future_agent import run_future_agent
# from src.agents.rental_agent import run_rental_agent
# from src.agents.advisor_agent import run_advisor_agent
# from src.services.prediction_service import predict_property_price
# from src.recommender.hybrid_recommender import apply_hybrid_ranking


# # =====================================================================
# # 1. CORE PIPELINE ENRICHMENT COMPONENT
# # =====================================================================
# def enrich_properties(selected_df: pd.DataFrame) -> pd.DataFrame:
#     """Enrich selected properties by running them through various analysis agents."""
#     df = selected_df.copy()
    
#     # MCP properties do not come from cosine similarity search; set default.
#     df["cosine_similarity"] = 1.0
#     df = apply_hybrid_ranking(df, intent={}, slider_weights=None)

#     # Define agents to run sequentially
#     # Format: (agent_function, name_for_logging/debugging)
#     agents = [
#         (run_analysis, "analysis"),
#         (run_negotiation_agent, "negotiation"),
#         (run_risk_agent, "risk"),
#         (run_future_agent, "future"),
#         (run_rental_agent, "rental")
#     ]

#     for agent_func, name in agents:
#         res = agent_func(df)
        
#         # Skip if result is None or an empty collection/dataframe
#         if res is None or (isinstance(res, (list, pd.DataFrame)) and len(res) == 0):
#             continue
            
#         # Convert to DataFrame if the agent returned a list of dicts
#         res_df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
        
#         # Merge the enrichment data
#         df = df.merge(res_df, on="id", how="left")

#     return df


# # =====================================================================
# # 2. MULTI-NODE INVESTMENT COMPARISON SERVICE
# # =====================================================================
# def run_mcp_comparison(property_ids: list[str]):
#     """Load selected properties, enrich them, and run comparison."""
#     # Filter master dataframe for the requested IDs
#     selected_df = master_df[master_df["id"].isin(property_ids)].copy()

#     # Need at least 2 properties to perform a comparison
#     if len(selected_df) < 2:
#         return selected_df, selected_df

#     # Enrich and compare
#     enriched_df = enrich_properties(selected_df)
#     return run_comparison(enriched_df)


# # =====================================================================
# # 3. ASSET RENTAL MATRIX SERVICE
# # =====================================================================
# def run_mcp_rental(property_ids: list[str]) -> pd.DataFrame:
#     """Isolates targets from master data pool and evaluates micro-rental performance yield."""
#     selected_df = master_df[master_df["id"].astype(str).isin([str(x) for x in property_ids])].copy()
#     if selected_df.empty:
#         return pd.DataFrame()
        
#     return run_rental_agent(selected_df)


# # =====================================================================
# # 4. FASTAPI PREDICTIVE VALUATION MODEL SERVICE (FIXED)
# # =====================================================================
# def run_mcp_prediction(property_ids: list[str]) -> pd.DataFrame:
#     """
#     Slices properties from master_df and runs external pipeline server forecasting calls.
#     Ensures strict fallback visibility and maps IDs safely to avoid pipeline leakage.
#     """
#     # Enforce clear element-wise string conversion to eliminate indexing mismatches
#     target_ids = [str(pid).strip().lower() for pid in property_ids]
    
#     # Slice matching properties out of master memory registry safely
#     selected_df = master_df[master_df["id"].astype(str).str.strip().str.lower().isin(target_ids)].copy()
#     results = []
    
#     if selected_df.empty:
#         print(f"⚠️ MCP Prediction Notice: Zero matching records identified for IDs: {property_ids}")
#         return pd.DataFrame(columns=["id", "location", "original_price", "predicted_price", "margin_diff"])
    
#     for _, row in selected_df.iterrows():
#         original_price = row.get("price", row.get("PRICE", 0))
#         p_id = row.get("id", row.get("ID", "Unknown"))
        
#         # Ensure the row object contains clean uppercase/lowercase tags expected by prediction_service
#         input_row = row.copy()
#         input_row["id"] = str(p_id).strip()
        
#         # Explicitly drop variants of price target from evaluation payloads to stop leakage
#         if "PRICE" in input_row.index:
#             input_row = input_row.drop("PRICE")
#         if "price" in input_row.index:
#             input_row = input_row.drop("price")
            
#         print(f"🔮 Dispatching MCP scoring payload to inference model engine for Asset ID: {p_id}...")
#         prediction_result = predict_property_price(input_row)
        
#         if prediction_result["success"]:
#             pred_data = prediction_result["prediction"]
#             # Extract from model output dictionary structure safely
#             predicted_price = pred_data.get("predicted_price")
            
#             # If the endpoint returned none or an empty dictionary fallback match string
#             if predicted_price is None:
#                 print(f"⚠️ Model API warning for {p_id}: 'predicted_price' key missing from return object. Triggering identity fallback.")
#                 predicted_price = original_price
#             else:
#                 print(f"✅ Prediction Successful for {p_id}: Model Output Price = ₹{predicted_price} Cr")
#         else:
#             # Fallback to base pricing metrics on pipeline system disconnects
#             print(f"❌ Model Pipeline Disconnect on Asset {p_id}: {prediction_result.get('error', 'Unknown Error State')}")
#             predicted_price = original_price
            
#         results.append({
#             "id": p_id,
#             "location": row.get("location", "Unknown"),
#             "original_price": float(original_price),
#             "predicted_price": round(float(predicted_price), 2),
#             "margin_diff": round(float(predicted_price) - float(original_price), 2)
#         })
        
#     return pd.DataFrame(results)


# # =====================================================================
# # 5. STRATEGIC NEGOTIATION & TALKING POINTS SERVICE
# # =====================================================================
# def run_mcp_negotiation(property_ids: list[str]) -> pd.DataFrame:
#     """Enriches context metrics and executes pricing leverage deduction matrices."""
#     selected_df = master_df[master_df["id"].astype(str).isin([str(x) for x in property_ids])].copy()
#     if selected_df.empty:
#         return pd.DataFrame()
        
#     enriched_df = enrich_properties(selected_df)
#     return run_negotiation_agent(enriched_df)


# # =====================================================================
# # 6. MARKET FAIR-VALUE AND VALIDATION SERVICE
# # =====================================================================
# def run_mcp_valuation(property_ids: list[str]) -> pd.DataFrame:
#     """Maps pricing distribution margins to clean market data structures."""
#     selected_df = master_df[master_df["id"].astype(str).isin([str(x) for x in property_ids])].copy()
#     if selected_df.empty:
#         return pd.DataFrame()
        
#     enriched_df = enrich_properties(selected_df)
#     analysis_list = run_analysis(enriched_df)
    
#     # Merge analytics lists back to tracking frames
#     analysis_df = pd.DataFrame(analysis_list)
#     return enriched_df[["id", "project_name", "price", "costpersqft"]].merge(analysis_df, on="id", how="left")


# # =====================================================================
# # 7. PORTFOLIO ADVISORY & SUITABILITY STACKING SERVICE
# # =====================================================================
# def run_mcp_advisor(property_ids: list[str]) -> pd.DataFrame:
#     """Combines high-level stacking matrix evaluations with base recommendations."""
#     # Fallback safety if the user attempts to trigger advice with only 1 staged item
#     if len(property_ids) < 2:
#         raw_df = master_df[master_df["id"].isin(property_ids)].copy()
#         raw_df = enrich_properties(raw_df)
#         compare_df = pd.DataFrame([{
#             "id": pid, "overall_score": 0.50, "verdict": "👍 Balanced", "comparison_reason": "Context incomplete"
#         } for pid in property_ids])
#     else:
#         raw_df, compare_df = run_mcp_comparison(property_ids)
        
#     advisor_df = run_advisor_agent(raw_df)
#     return advisor_df.merge(compare_df[["id", "overall_score", "verdict"]], on="id", how="left")



#============================================================================================
#some improvments in above code 

# # =====================================================================
# # src/services/mcp_real_estate_service.py (PRISTINE DATATYPES & THREADED)
# # =====================================================================

# import pandas as pd
# from threading import Lock
# from src.data.data_store import master_df

# # Connection to standalone comparison engine bridge
# from src.services.comparison_service import run_comparison

# # Legacy agent core execution engines
# from src.agents.analysis_agent import run_analysis
# from src.agents.negotiation_agent import run_negotiation_agent
# from src.agents.risk_agent import run_risk_agent
# from src.agents.future_agent import run_future_agent
# from src.agents.rental_agent import run_rental_agent
# from src.agents.advisor_agent import run_advisor_agent
# from src.services.prediction_service import predict_property_price
# from src.recommender.hybrid_recommender import apply_hybrid_ranking

# # ---------------------------------------------------------------------
# # 📍 THREAD-SAFE CONCURRENCY GUARDED MEMORY CACHE REGISTER
# # ---------------------------------------------------------------------
# ENRICHMENT_CACHE = {}
# CACHE_LOCK = Lock()


# def clear_enrichment_cache():
#     """
#     Manually invalidates the runtime memory storage layer cleanly across threads.
#     Essential for hot-reloading changes to agent files during active development.
#     """
#     global ENRICHMENT_CACHE
#     with CACHE_LOCK:
#         ENRICHMENT_CACHE.clear()
#     print("🧹 MCP enrichment cache cleared successfully across active threads!")


# def enrich_properties(selected_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     enrich_properties() takes the original property dataframe and enriches it by adding important calculated columns from the ranking, analysis, risk, 
#     future growth, rental, and negotiation modules, returning a single dataframe containing both the original property data and all generated insights.
#     """
#     if selected_df.empty:
#         return selected_df

#     df = selected_df.copy()
    
#     # MCP does not perform similarity search like the recommendation engine.
#     # Set cosine_similarity = 1.0 for all properties so we can reuse the
#     # existing hybrid ranking pipeline and generate ranking-related fields.
#     df["cosine_similarity"] = 1.0
#     df = apply_hybrid_ranking(df, intent={}, slider_weights=None)

#     # Order arranged so negotiation can parse calculated risk/growth/valuation outputs
#     agents = [
#         (run_analysis, "analysis"),
#         (run_risk_agent, "risk"),
#         (run_future_agent, "future"),
#         (run_rental_agent, "rental"),
#         (run_negotiation_agent, "negotiation")  # Executes last with full contextual layout
#     ]

#     for agent_func, name in agents:
#         try:
#             res = agent_func(df)
#             if res is None or (isinstance(res, (list, pd.DataFrame)) and len(res) == 0):
#                 continue
                
#             res_df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
            
#             # Prevent schema fragmentation on multiple merge iterations
#             overlap_cols = [c for c in res_df.columns if c in df.columns and c != "id"]
#             if overlap_cols:
#                 res_df = res_df.drop(columns=overlap_cols)
                
#             df = df.merge(res_df, on="id", how="left")
#         except Exception as e:
#             print(f"⚠️ Bypass Warning: Agent [{name}] failed baseline execution matrix: {str(e)}")
#             continue

#     return df


# def get_cached_enrichment(property_ids: list[str]) -> pd.DataFrame:
    
#     """If a property's enriched data is already in memory, use it. Otherwise, calculate it once, store it in cache, and reuse it later."""

#     global ENRICHMENT_CACHE
    
#     # Standardize incoming elements cleanly
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
    
#     cached_frames = []
#     missing_ids = []
    
#     # 1. Thread-safe read checkpoint mapping processed memory blocks
#     with CACHE_LOCK:
#         for pid in target_ids:
#             if pid in ENRICHMENT_CACHE:
#                 cached_frames.append(ENRICHMENT_CACHE[pid])
#             else:
#                 missing_ids.append(pid)
            
#     # 2. Compute missing assets exclusively on cache misses
#     if missing_ids:
#         print(f"🧩 Cache Miss: Slicing and orchestrating enrichment for {len(missing_ids)} unique properties...")
        
#         raw_missing_df = master_df[master_df["id"].astype(str).str.strip().isin(missing_ids)].copy()
        
#         if not raw_missing_df.empty:
#             enriched_missing_df = enrich_properties(raw_missing_df)
            
#             # FIXED: Restored clean iterrows loop with strict dataframe transposition to protect types
#             with CACHE_LOCK:
#                 for _, row in enriched_missing_df.iterrows():
#                     pid_key = str(row["id"]).strip()
                    
#                     # Transpose cleanly into a single-row tracking frame, locking down explicit dtypes
#                     single_row_df = row.to_frame().T
                    
#                     # Store exact schema block to guarantee clean future frame concatenations
#                     ENRICHMENT_CACHE[pid_key] = single_row_df
#                     cached_frames.append(single_row_df)

#     # 3. Secure type alignment through uniform master axis concatenation
#     if not cached_frames:
#         return pd.DataFrame()
        
#     return pd.concat(cached_frames, ignore_index=True)


# # =====================================================================
# # 2. MULTI-NODE INVESTMENT COMPARISON SERVICE
# # =====================================================================
# def run_mcp_comparison(property_ids: list[str]):
#     """This function fetches all enriched data(enriched_df)i.e row data for the selected properties and then sends that data to the comparison model to calculate rankings and scores."""
#     enriched_df = get_cached_enrichment(property_ids)

#     if len(enriched_df) < 2:
#         return enriched_df, enriched_df

#     return run_comparison(enriched_df)


# # =====================================================================
# # 3. ASSET RENTAL MATRIX SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_rental(property_ids: list[str]) -> pd.DataFrame:
#     """Extracts pre-computed asset yield variables straight out of cached states."""
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     rental_schema = [
#         "id",
#         "monthly_rent_estimate",
#         "annual_rent",
#         "rental_yield_percent",
#         "demand_level",
#         "investment_rating",
#         "rental_strategy"
#     ]
    
#     valid_cols = [c for c in rental_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 4. FASTAPI PREDICTIVE VALUATION MODEL SERVICE
# # =====================================================================
# def run_mcp_prediction(property_ids: list[str]) -> pd.DataFrame:
#     """
#     Slices targets using optimized cache tracking and dispatches data 
#     directly to inference prediction models safely.
#     """
#     enriched_df = get_cached_enrichment(property_ids)
#     results = []
    
#     if enriched_df.empty:
#         return pd.DataFrame(columns=["id", "location", "original_price", "predicted_price", "margin_diff"])
    
#     for _, row in enriched_df.iterrows():
#         original_price = row.get("price", row.get("PRICE", 0))
#         p_id = row.get("id", row.get("ID", "Unknown"))
        
#         input_row = row.copy()
#         input_row["id"] = str(p_id).strip()
        
#         # Eliminate data leakage points prior to inference
#         for price_key in ["PRICE", "price"]:
#             if price_key in input_row.index:
#                 input_row = input_row.drop(price_key)
                
#         print(f"🔮 Dispatching cached parameters to FastAPI endpoint for Asset ID: {p_id}...")
#         prediction_result = predict_property_price(input_row)
        
#         if prediction_result["success"]:
#             pred_data = prediction_result["prediction"]
#             predicted_price = pred_data.get("predicted_price")
#             if predicted_price is None:
#                 predicted_price = original_price
#         else:
#             predicted_price = original_price
            
#         results.append({
#             "id": p_id,
#             "location": row.get("location", "Unknown"),
#             "original_price": float(original_price),
#             "predicted_price": round(float(predicted_price), 2),
#             "margin_diff": round(float(predicted_price) - float(original_price), 2)
#         })
        
#     return pd.DataFrame(results)


# # =====================================================================
# # 5. STRATEGIC NEGOTIATION & TALKING POINTS SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_negotiation(property_ids: list[str]) -> pd.DataFrame:
#     """Bypasses rule agent computation passes by returning compiled metric keys directly."""
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     negotiation_schema = [
#         "id",
#         "negotiation_power",
#         "suggested_discount_percent",
#         "target_price",
#         "price_position",
#         "strategy",
#         "talking_points"
#     ]
    
#     valid_cols = [c for c in negotiation_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 6. MARKET FAIR-VALUE AND VALIDATION SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_valuation(property_ids: list[str]) -> pd.DataFrame:
#     """Bypasses re-analysis by pulling benchmark variance indicators from cache."""
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     valuation_schema = [
#         "id",
#         "project_name",
#         "price",
#         "costpersqft",
#         "analysis_flag",
#         "analysis_msg",
#         "analysis_severity"
#     ]
    
#     valid_cols = [c for c in valuation_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 7. PORTFOLIO ADVISORY & SUITABILITY STACKING SERVICE
# # =====================================================================
# def run_mcp_advisor(property_ids: list[str]) -> pd.DataFrame:
#     """Uses cached tables to render direct decision portfolio summaries."""
#     enriched_df = get_cached_enrichment(property_ids)
    
#     if len(property_ids) < 2:
#         compare_df = pd.DataFrame([{
#             "id": pid, "overall_score": 0.50, "verdict": "👍 Balanced", "comparison_reason": "Context incomplete"
#         } for pid in property_ids])
#     else:
#         _, compare_df = run_mcp_comparison(property_ids)
        
#     advisor_df = run_advisor_agent(enriched_df)
    
#     # Drop colliding metadata elements if present prior to joining fields
#     dup_cols = [c for c in compare_df.columns if c in advisor_df.columns and c != "id"]
#     if dup_cols:
#         compare_df = compare_df.drop(columns=dup_cols)
        
#     return advisor_df.merge(compare_df[["id", "overall_score", "verdict"]], on="id", how="left")


#======================================================================================================================================================================================

# # =====================================================================
# # src/services/mcp_real_estate_service.py (PRISTINE DATATYPES & THREADED)
# # =====================================================================

# import pandas as pd
# from threading import Lock
# from src.data.data_store import master_df

# # Connection to standalone comparison engine bridge
# from src.services.comparison_service import run_comparison

# # Legacy agent core execution engines
# from src.agents.analysis_agent import run_analysis
# from src.agents.negotiation_agent import run_negotiation_agent
# from src.agents.risk_agent import run_risk_agent
# from src.agents.future_agent import run_future_agent
# from src.agents.rental_agent import run_rental_agent
# from src.agents.advisor_agent import run_advisor_agent
# from src.services.prediction_service import predict_property_price
# from src.recommender.hybrid_recommender import apply_hybrid_ranking

# from src.utils.rent_utils import calculate_rent

# # ---------------------------------------------------------------------
# # 📍 THREAD-SAFE CONCURRENCY GUARDED MEMORY CACHE REGISTER
# # ---------------------------------------------------------------------
# ENRICHMENT_CACHE = {}
# CACHE_LOCK = Lock()


# def clear_enrichment_cache():
#     """
#     Manually invalidates the runtime memory storage layer cleanly across threads.
#     Essential for hot-reloading changes to agent files during active development.
#     """
#     global ENRICHMENT_CACHE
#     with CACHE_LOCK:
#         ENRICHMENT_CACHE.clear()
#     print("🧹 MCP enrichment cache cleared successfully across active threads!")


# def enrich_properties(selected_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     enrich_properties() takes the original property dataframe and enriches it by adding important calculated columns from the ranking, analysis, risk, 
#     future growth, rental, and negotiation modules, returning a single dataframe containing both the original property data and all generated insights.
#     """
#     if selected_df.empty:
#         return selected_df

#     df = selected_df.copy()
    
#     # MCP does not perform similarity search like the recommendation engine.
#     # Set cosine_similarity = 1.0 for all properties so we can reuse the
#     # existing hybrid ranking pipeline and generate ranking-related fields.
#     df["cosine_similarity"] = 1.0
#     df = apply_hybrid_ranking(
#         similar_df=df,
#         intent_weights={},
#         slider_weights=None
#     )

#     df[["estimated_rent_min", "estimated_rent_max"]] = df.apply(
#         lambda row: pd.Series(calculate_rent(row)),
#         axis=1
#     )

#     print("3"*50)
#     print("df columns list full", df.columns.tolist())
#     print("3"*50)

#     # Order arranged so negotiation can parse calculated risk/growth/valuation outputs
#     agents = [
#         (run_analysis, "analysis"),
#         (run_risk_agent, "risk"),
#         (run_future_agent, "future"),
#         (run_rental_agent, "rental"),
#         (run_negotiation_agent, "negotiation")  
#     ]

#     for agent_func, name in agents:
#         try:
#             res = agent_func(df)
#             #Skip empty results
#             if res is None or (isinstance(res, (list, pd.DataFrame)) and len(res) == 0):
#                 continue
                
#             res_df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
            
#             # Prevent duplicate columns
#             overlap_cols = [c for c in res_df.columns if c in df.columns and c != "id"]
#             if overlap_cols:
#                 res_df = res_df.drop(columns=overlap_cols) # drop duplicate columns
                
#             df = df.merge(res_df, on="id", how="left") 
#         except Exception as e:
#             print(f"⚠️ Bypass Warning: Agent [{name}] failed baseline execution matrix: {str(e)}")
#             continue

#     return df


# def get_cached_enrichment(property_ids: list[str]) -> pd.DataFrame:
    
#     """
#     If a property's enriched data is already in memory, use it. Otherwise, calculate it once, store it in cache, and reuse it later.
#     also this function returns all properties requested by the chatbot 
#     """

#     global ENRICHMENT_CACHE
    
#     # Standardize incoming elements cleanly
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
    
#     cached_frames = [] #Properties already enriched
#     missing_ids = [] #Properties not yet enriched
    
#     # 1. Thread-safe read checkpoint mapping processed memory blocks
#     with CACHE_LOCK:
#         for pid in target_ids:
#             if pid in ENRICHMENT_CACHE:
#                 cached_frames.append(ENRICHMENT_CACHE[pid])
#             else:
#                 missing_ids.append(pid)
            
#     # 2. Compute missing assets exclusively on cache misses
#     if missing_ids:
#         print(f"🧩 Cache Miss: Slicing and orchestrating enrichment for {len(missing_ids)} unique properties...")
        
#         #master_df is final_combined_mcp_data from data_store.py
#         raw_missing_df = master_df[master_df["id"].astype(str).str.strip().isin(missing_ids)].copy() # Ensure the ID slicing is clean and matches the standardized keys used in cache mapping
        
#         if not raw_missing_df.empty:
#             # Run all enrichment agents on only the uncached properties
#             # (risk, future growth, rental, negotiation, etc.)
#             enriched_missing_df = enrich_properties(raw_missing_df) 
            
#             # Save newly enriched properties into cache
#             with CACHE_LOCK:
#                 for _, row in enriched_missing_df.iterrows():
#                     pid_key = str(row["id"]).strip()
                    
#                     # Convert row → dataframe
#                     single_row_df = row.to_frame().T
                    
#                     # Store enriched property dataframe in cache
#                     ENRICHMENT_CACHE[pid_key] = single_row_df

#                     # Also add it to the result list that will be returned
#                     cached_frames.append(single_row_df)

#     # If no properties were found (cached or newly enriched), return an empty dataframe.
#     if not cached_frames:
#         return pd.DataFrame()

#     # Combine all cached and newly enriched property dataframes into one final dataframe and return it.        
#     return pd.concat(cached_frames, ignore_index=True)


# # =====================================================================
# # 2. MULTI-NODE INVESTMENT COMPARISON SERVICE
# # =====================================================================
# def run_mcp_comparison(property_ids: list[str]):
#     """This function fetches all enriched data(enriched_df)i.e row data for the selected properties and then sends that data to the comparison model to calculate rankings and scores."""
#     enriched_df = get_cached_enrichment(property_ids)

#     if len(enriched_df) < 2:
#         return enriched_df, enriched_df

#     return run_comparison(enriched_df)


# # =====================================================================
# # 3. ASSET RENTAL MATRIX SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_rental(property_ids: list[str]) -> pd.DataFrame:
#     """
#     Get rental analysis for requested properties.
#     Uses cached enriched data and returns only
#     rental-related columns without re-running
#     the rental agent.
#     """
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     rental_schema = [
#         "id",
#         "monthly_rent_estimate",
#         "annual_rent",
#         "rental_yield_percent",
#         "demand_level",
#         "investment_rating",
#         "rental_strategy"
#     ]
    
#     valid_cols = [c for c in rental_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 4. FASTAPI PREDICTIVE VALUATION MODEL SERVICE
# # =====================================================================
# def run_mcp_prediction(property_ids: list[str]) -> pd.DataFrame:
#     """
#     Slices targets using optimized cache tracking and dispatches data 
#     directly to inference prediction models safely.
#     """
#     enriched_df = get_cached_enrichment(property_ids)
#     results = []
    
#     if enriched_df.empty:
#         return pd.DataFrame(columns=["id", "location", "original_price", "predicted_price", "margin_diff"])
    
#     for _, row in enriched_df.iterrows():
#         original_price = row.get("price", row.get("PRICE", 0))
#         p_id = row.get("id", row.get("ID", "Unknown"))
        
#         input_row = row.copy()
#         input_row["id"] = str(p_id).strip()
        
#         # Eliminate data leakage points prior to inference
#         for price_key in ["PRICE", "price"]:
#             if price_key in input_row.index:
#                 input_row = input_row.drop(price_key)
                
#         print(f"🔮 Dispatching cached parameters to FastAPI endpoint for Asset ID: {p_id}...")

#         # print("\n========== MCP ROW ==========")
#         # print(row)
#         # print("=============================\n")

#         # Call prediction model.
#         prediction_result = predict_property_price(input_row)
#         # print("\n========== PREDICTION RESULT ==========")
#         # print(prediction_result)
#         # print("=======================================\n")
        
#         # If prediction fails, mark as "Prediction Failed".
#         # If prediction value is missing, mark as "Prediction Missing".
#         # Otherwise use the predicted price for further calculations.
#         if not prediction_result["success"]:
#             results.append({
#                 "id": p_id,
#                 "location": row.get("location", "Unknown"),
#                 "original_price": float(original_price),
#                 "predicted_price": None,
#                 "margin_diff": None,
#                 "status": "Prediction Failed"
#             })
#             continue

#         pred_data = prediction_result["prediction"]
#         predicted_price = pred_data.get("predicted_price")

#         if predicted_price is None:
#             results.append({
#                 "id": p_id,
#                 "location": row.get("location", "Unknown"),
#                 "original_price": float(original_price),
#                 "predicted_price": None,
#                 "margin_diff": None,
#                 "status": "Prediction Missing"
#             })
#             continue
            
#         results.append({
#             "id": p_id,
#             "location": row.get("location", "Unknown"),
#             "original_price": float(original_price),
#             "predicted_price": round(float(predicted_price), 2),
#             "margin_diff": round(float(predicted_price) - float(original_price), 2)
#         })
        
#     return pd.DataFrame(results)


# # =====================================================================
# # 5. STRATEGIC NEGOTIATION & TALKING POINTS SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_negotiation(property_ids: list[str]) -> pd.DataFrame:
#     """Bypasses rule agent computation passes by returning compiled metric keys directly."""
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     negotiation_schema = [
#         "id",
#         "negotiation_power",
#         "suggested_discount_percent",
#         "target_price",
#         "price_position",
#         "strategy",
#         "talking_points"
#     ]
    
#     valid_cols = [c for c in negotiation_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 6. MARKET FAIR-VALUE AND VALIDATION SERVICE (ZERO RE-COMPUTATION)
# # =====================================================================
# def run_mcp_valuation(property_ids: list[str]) -> pd.DataFrame:
#     """Bypasses re-analysis by pulling benchmark variance indicators from cache."""
#     enriched_df = get_cached_enrichment(property_ids)
#     if enriched_df.empty:
#         return pd.DataFrame()
        
#     valuation_schema = [
#         "id",
#         "project_name",
#         "price",
#         "costpersqft",
#         "analysis_flag",
#         "analysis_msg",
#         "analysis_severity"
#     ]
    
#     valid_cols = [c for c in valuation_schema if c in enriched_df.columns]
#     return enriched_df[valid_cols]


# # =====================================================================
# # 7. PORTFOLIO ADVISORY & SUITABILITY STACKING SERVICE
# # =====================================================================
# def run_mcp_advisor(property_ids: list[str]) -> pd.DataFrame:
#     """Uses cached tables to render direct decision portfolio summaries."""
#     enriched_df = get_cached_enrichment(property_ids)
    
#     if len(property_ids) < 2:
#         compare_df = pd.DataFrame([{
#             "id": pid, "overall_score": 0.50, "verdict": "👍 Balanced", "comparison_reason": "Context incomplete"
#         } for pid in property_ids])
#     else:
#         _, compare_df = run_mcp_comparison(property_ids)
        
#     advisor_df = run_advisor_agent(enriched_df)
    
#     # Drop colliding metadata elements if present prior to joining fields
#     dup_cols = [c for c in compare_df.columns if c in advisor_df.columns and c != "id"]
#     if dup_cols:
#         compare_df = compare_df.drop(columns=dup_cols)
        
#     return advisor_df.merge(compare_df[["id", "overall_score", "verdict"]], on="id", how="left")


#==============================================================================================


# =====================================================================
# src/services/mcp_real_estate_service.py (PRISTINE DATATYPES & THREADED)
# =====================================================================

import pandas as pd
from threading import Lock
from src.data.data_store import master_df

# Connection to standalone comparison engine bridge
from src.services.comparison_service import run_comparison

# Legacy agent core execution engines
from src.agents.analysis_agent import run_analysis
from src.agents.negotiation_agent import run_negotiation_agent
from src.agents.risk_agent import run_risk_agent
from src.agents.future_agent import run_future_agent
from src.agents.rental_agent import run_rental_agent
from src.agents.advisor_agent import run_advisor_agent
from src.services.prediction_service import predict_property_price
from src.recommender.hybrid_recommender import apply_hybrid_ranking

from src.utils.rent_utils import calculate_rent

# ---------------------------------------------------------------------
# 📍 THREAD-SAFE CONCURRENCY GUARDED MEMORY CACHE REGISTER
# ---------------------------------------------------------------------
ENRICHMENT_CACHE = {}  # Stores processed property data so it doesn't need to be recalculated.
CACHE_LOCK = Lock()   # Ensures cache updates are safe when multiple users/threads access it simultaneously.


def clear_enrichment_cache():
    """
    Manually invalidates the runtime memory storage layer cleanly across threads.
    Essential for hot-reloading changes to agent files during active development.
    """
    print("☑️ clear_enrichment_cache executed")
    global ENRICHMENT_CACHE
    with CACHE_LOCK:
        ENRICHMENT_CACHE.clear()
    print("🧹 MCP enrichment cache cleared successfully across active threads!")


def enrich_properties(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a single dataframe containing both the original property data
    and all agent-generated insights.
    """
    print("☑️ enrich_properties executed")
    if selected_df.empty:
        return selected_df

    df = selected_df.copy()
    
    # MCP does not perform similarity search like the recommendation engine.
    # Set cosine_similarity = 1.0 for all properties so we can reuse the
    # hybrid ranking pipeline and generate ranking-related fields.

    # MCP does not use chat preferences or UI slider weights.
    # apply_hybrid_ranking() will therefore fall back to the default
    # baseline ranking weights defined in hybrid_recommender.py.

    # MCP uses apply_hybrid_ranking() only to generate standardized scoring
    # fields (price_score, area_score, weighted_score, hybrid_score, etc.)
    # which are later consumed by the analysis, risk, rental, and negotiation agents.
    df["cosine_similarity"] = 1.0


    df = apply_hybrid_ranking(
        similar_df=df,
        intent_weights={},
        slider_weights=None
    )

    df[["estimated_rent_min", "estimated_rent_max"]] = df.apply(
        lambda row: pd.Series(calculate_rent(row)),
        axis=1
    )

    print("3"*50)
    print("df columns list full", df.columns.tolist())
    print("3"*50)

    # Order arranged so negotiation can parse calculated risk/growth/valuation outputs
    agents = [
        (run_analysis, "analysis"),
        (run_risk_agent, "risk"),
        (run_future_agent, "future"),
        (run_rental_agent, "rental"),
        (run_negotiation_agent, "negotiation")  
    ]

    # Merge the new columns generated by each agent into the main property dataframe.
    for agent_func, name in agents: # Execute each enrichment agent one by on
        try:
            res = agent_func(df) # Execute the current agent using the enriched property dataframe.
            # Skip agents that return no data.
            if res is None or (isinstance(res, (list, pd.DataFrame)) and len(res) == 0):
                continue
                
            res_df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res) # Convert agent output to a dataframe if it isn't one already.
            
            # Identify columns that already exist in the main dataframe
            # (except the 'id' column used for merging).
            overlap_cols = [c for c in res_df.columns if c in df.columns and c != "id"]
            if overlap_cols: # Remove duplicate columns from the agent output before merging.
                res_df = res_df.drop(columns=overlap_cols) 
                
            df = df.merge(res_df, on="id", how="left")  # Merge the agent-generated insights into the main dataframe using property ID as the common key.

        except Exception as e:
            # If one agent fails, log the error and continue with
            # the remaining agents instead of stopping the pipeline.
            print(f"⚠️ Bypass Warning: Agent [{name}] failed baseline execution matrix: {str(e)}")
            continue

    return df # Return the final enriched dataframe containing the original property data plus all successfully generated agent insights.


def get_cached_enrichment(property_ids: list[str]) -> pd.DataFrame:
    
    """
    If a property's enriched data(Original property data + all calculated insights added by our agents.) is already in memory, use it. Otherwise, 
    calculate it once, store it in cache, and reuse it later.
    also this get_cached_enrichment function returns all properties requested by the chatbot 
    """
    print("☑️ get_cached_enrichment executed")
    global ENRICHMENT_CACHE
    
    # Standardize incoming elements cleanly
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    
    cached_frames = [] #Properties already enriched
    missing_ids = [] #Properties not yet enriched
    
    # 1. Check which properties are already cached and which need enrichment.
    with CACHE_LOCK:
        for pid in target_ids:
            if pid in ENRICHMENT_CACHE:
                cached_frames.append(ENRICHMENT_CACHE[pid])
            else:
                missing_ids.append(pid)

    print(
        f"📦 CACHE HIT={len(cached_frames)} "
        f"CACHE MISS={len(missing_ids)} "
        f"REQUESTED={len(target_ids)}"
    )

    print("TARGET IDS:", target_ids[:10])
            
    # 2. Compute missing assets exclusively on cache misses
    if missing_ids:
        print(f"🧩 Cache Miss: Slicing and orchestrating enrichment for {len(missing_ids)} unique properties...")
        
        #master_df is final_combined_mcp_data from data_store.py
        raw_missing_df = master_df[master_df["id"].astype(str).str.strip().isin(missing_ids)].copy() # Ensure the ID slicing is clean and matches the standardized keys used in cache mapping
        
        if not raw_missing_df.empty:
            # Returns a single dataframe containing both the original property data and all agent-generated insights.
            # for properties that are not already cached.
            enriched_missing_df = enrich_properties(raw_missing_df)  
            
            # Save newly enriched properties into cache
            with CACHE_LOCK:
                for _, row in enriched_missing_df.iterrows():
                    pid_key = str(row["id"]).strip()
                    
                    # Convert row → dataframe
                    single_row_df = row.to_frame().T
                    
                    # Store enriched property dataframe in cache
                    ENRICHMENT_CACHE[pid_key] = single_row_df

                    # Also add it to the result list that will be returned
                    cached_frames.append(single_row_df)

    # If no properties were found (cached or newly enriched), return an empty dataframe.
    if not cached_frames:
        return pd.DataFrame()

    # Combine all cached and newly enriched property dataframes into one final dataframe and return it.        
    return pd.concat(cached_frames, ignore_index=True)


# =====================================================================
# 2. MULTI-NODE INVESTMENT COMPARISON SERVICE
# =====================================================================
def run_mcp_comparison(property_ids: list[str]):
    """
    Retrieves enriched property data(Original property data + all calculated insights added by our agents.) and runs the comparison model
    to generate rankings and investment scores.
    """
    print("☑️ run_mcp_comparison executed")
    enriched_df = get_cached_enrichment(property_ids)

    # Comparison requires at least 2 properties.
    if len(enriched_df) < 2:
        return enriched_df, enriched_df

    return run_comparison(enriched_df)


# =====================================================================
# 3. ASSET RENTAL MATRIX SERVICE (ZERO RE-COMPUTATION)
# =====================================================================
def run_mcp_rental(property_ids: list[str]) -> pd.DataFrame:
    """
    Get rental analysis for requested properties.
    Uses cached enriched data and returns only
    rental-related columns without re-running
    the rental agent.
    """
    print("☑️ run_mcp_rental executed")
    enriched_df = get_cached_enrichment(property_ids)
    if enriched_df.empty:
        return pd.DataFrame()
        
    rental_schema = [
        "id",
        "monthly_rent_estimate",
        "annual_rent",
        "rental_yield_percent",
        "demand_level",
        "investment_rating",
        "rental_strategy"
    ]
    
    valid_cols = [c for c in rental_schema if c in enriched_df.columns]
    return enriched_df[valid_cols]


# =====================================================================
# 4. FASTAPI PREDICTIVE VALUATION MODEL SERVICE
# =====================================================================
def run_mcp_prediction(property_ids: list[str]) -> pd.DataFrame:
    """
    Slices targets using optimized cache tracking and dispatches data 
    directly to inference prediction models safely.
    """
    print("☑️ run_mcp_prediction executed")
    enriched_df = get_cached_enrichment(property_ids)
    results = []
    
    if enriched_df.empty:
        return pd.DataFrame(columns=["id", "location", "original_price", "predicted_price", "margin_diff"])
    
    for _, row in enriched_df.iterrows():
        original_price = row.get("price", row.get("PRICE", 0))
        p_id = row.get("id", row.get("ID", "Unknown"))
        
        input_row = row.copy()
        input_row["id"] = str(p_id).strip()
        
        # Remove the actual property price before sending the property data to the prediction model,to avoid data leakage
        for price_key in ["PRICE", "price"]:
            if price_key in input_row.index:
                input_row = input_row.drop(price_key)
                
        print(f"🔮 Dispatching cached parameters to FastAPI endpoint for Asset ID: {p_id}...")

        # print("\n========== MCP ROW ==========")
        # print(row)
        # print("=============================\n")

        # Call prediction model.
        prediction_result = predict_property_price(input_row)
        # print("\n========== PREDICTION RESULT ==========")
        # print(prediction_result)
        # print("=======================================\n")
        
        # If the prediction request fails, record the failure and continue with the next property.
        if not prediction_result or not prediction_result.get("success"):
            results.append({
                "id": p_id,
                "location": row.get("location", "Unknown"),
                "original_price": float(original_price),
                "predicted_price": None,
                "margin_diff": None,
                "status": "Prediction Failed"
            })
            continue
        
        # Extract the predicted price returned by the prediction model.
        pred_data = prediction_result["prediction"]
        predicted_price = pred_data.get("predicted_price")

        # The prediction process succeeded, but the predicted price is missing, so record prediction and continue with the next property.
        if predicted_price is None:
            results.append({
                "id": p_id,
                "location": row.get("location", "Unknown"),
                "original_price": float(original_price),
                "predicted_price": None,
                "margin_diff": None,
                "status": "Prediction Missing"
            })
            continue
        
        # Save the successful prediction along with the price difference.
        results.append({
            "id": p_id,
            "location": row.get("location", "Unknown"),
            "original_price": float(original_price),
            "predicted_price": round(float(predicted_price), 2),
            "margin_diff": round(float(predicted_price) - float(original_price), 2)
        })
    
    # Return prediction results for all requested properties as a DataFrame.
    return pd.DataFrame(results)


# =====================================================================
# 5. STRATEGIC NEGOTIATION & TALKING POINTS SERVICE (ZERO RE-COMPUTATION)
# =====================================================================
def run_mcp_negotiation(property_ids: list[str]) -> pd.DataFrame:
    """Bypasses rule agent computation passes by returning compiled metric keys directly."""
    print("☑️ run_mcp_negotiation executed")
    enriched_df = get_cached_enrichment(property_ids)
    if enriched_df.empty:
        return pd.DataFrame()
        
    negotiation_schema = [
        "id",
        "negotiation_power",
        "suggested_discount_percent",
        "target_price",
        "price_position",
        "strategy",
        "talking_points"
    ]
    
    valid_cols = [c for c in negotiation_schema if c in enriched_df.columns]
    return enriched_df[valid_cols]


# =====================================================================
# 6. MARKET FAIR-VALUE AND VALIDATION SERVICE (ZERO RE-COMPUTATION)
# =====================================================================
def run_mcp_valuation(property_ids: list[str]) -> pd.DataFrame:
    """Bypasses re-analysis by pulling benchmark variance indicators from cache."""
    print("☑️ run_mcp_valuation executed")
    enriched_df = get_cached_enrichment(property_ids)
    if enriched_df.empty:
        return pd.DataFrame()
        
    valuation_schema = [
        "id",
        "project_name",
        "price",
        "costpersqft",
        "analysis_flag",
        "analysis_msg",
        "analysis_severity"
    ]
    
    valid_cols = [c for c in valuation_schema if c in enriched_df.columns]
    return enriched_df[valid_cols]


# =====================================================================
# 7. PORTFOLIO ADVISORY & SUITABILITY STACKING SERVICE
# =====================================================================
def run_mcp_advisor(property_ids: list[str]) -> pd.DataFrame:
    """Uses cached tables to render direct decision portfolio summaries."""
    print("☑️ run_mcp_advisor executed")
    enriched_df = get_cached_enrichment(property_ids)
    
    # If only one property is selected, create a default comparison result since comparison requires at least two properties.
    if len(property_ids) < 2:
        compare_df = pd.DataFrame([{
            "id": pid, "overall_score": 0.50, "verdict": "👍 Balanced", "comparison_reason": "Context incomplete"
        } for pid in property_ids])
    else:
        _, compare_df = run_mcp_comparison(property_ids)
        
    advisor_df = run_advisor_agent(enriched_df) # Extract advisor-specific columns from the enriched property data.
    
    # Remove duplicate columns before merging the comparison results.
    dup_cols = [c for c in compare_df.columns if c in advisor_df.columns and c != "id"]
    if dup_cols:
        compare_df = compare_df.drop(columns=dup_cols) 

    # Merge advisor insights with the comparison score and verdict.        
    return advisor_df.merge(compare_df[["id", "overall_score", "verdict"]], on="id", how="left")