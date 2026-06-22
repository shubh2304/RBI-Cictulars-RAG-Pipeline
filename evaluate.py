import time
import os
from database.connection import get_connection
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker

# Curated benchmark dataset of actual compliance queries mapped to document identifiers and sections
BENCHMARK_DATASET = [
    {
        "query": "What is the limit for collateral-free agricultural loans?",
        "expected_filename": "RBI-2024-2025-96_712202414454187.pdf",
        "expected_section": "Credit Flow to Agriculture"
    },
    {
        "query": "What are the priority sector lending targets for Regional Rural Banks?",
        "expected_filename": "34MD27062019.pdf",
        "expected_section": "MASTER DIRECTION-REGIONAL RURAL BANKS"
    },
    {
        "query": "Are retail and wholesale trade included in the MSME sector for priority sector lending?",
        "expected_filename": "56MD24072017E50D0ED63F9B4414AA756FF0FC72FB66.pdf",
        "expected_section": "3.1"
    },
    {
        "query": "What are the eligibility requirements for Perpetual Debt Instruments in Basel III?",
        "expected_filename": "BASEL III CAPITAL FRAMEWORK.pdf",
        "expected_section": "Basel III Capital Regulations"
    },
    {
        "query": "What is the Block Level Bankers Committee constitution and frequency under the Lead Bank Scheme?",
        "expected_filename": "Master-circular-on-lead-bank-scheme.pdf",
        "expected_section": "2.1"
    },
    {
        "query": "What is the modified interest subvention scheme for short-term crop loans?",
        "expected_filename": "KCC Modified.pdf",
        "expected_section": "Kisan Credit Card (KCC) Scheme"
    },
    {
        "query": "What are the credit facilities provided to Scheduled Castes and Scheduled Tribes?",
        "expected_filename": "dri rbi.pdf",
        "expected_section": "CREDIT FACILITIES TO SCHEDULED CASTES"
    },
    {
        "query": "How is income and indebtedness assessed at the household level for microfinance loans?",
        "expected_filename": "RFML30012025.pdf",
        "expected_section": "Q 2."
    },
    {
        "query": "What are the risk weights permitted for claims guaranteed by CGTMSE?",
        "expected_filename": "Sep72022_RBI_RBI_Review_of_Prudential_Norms_in_regard_to_Risk_Weights_for_Exposures_guaranteed_by_Credit_Guarantee_Schemes.PDF",
        "expected_section": "Review of Prudential Norms"
    },
    {
        "query": "What are the interest rate guidelines for loans under DAY-NRLM?",
        "expected_filename": "DAY NRLMS.pdf",
        "expected_section": "Master Circular"
    }
]

def run_evaluation():
    print("======================================================================")
    print("                  RBI RAG SYSTEM EVALUATION ENGINE                    ")
    print("======================================================================")
    
    hybrid_retriever = HybridRetriever()
    
    total_queries = len(BENCHMARK_DATASET)
    reciprocal_ranks = []
    
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    
    latencies = []
    
    print(f"Loaded {total_queries} evaluation queries. Running retrieval test...")
    print("-" * 70)
    
    for idx, qa in enumerate(BENCHMARK_DATASET):
        query = qa["query"]
        expected_file = qa["expected_filename"]
        
        start_time = time.time()
        # Retrieve candidates via hybrid search (Top 15)
        candidates = hybrid_retriever.search(query, top_k=15)
        # Rerank down to Top 5
        reranked = Reranker.rerank(query, candidates, top_k=5)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        # Check ranks
        rank_found = -1
        for rank, chunk in enumerate(reranked):
            # Check if expected document is matched by filename
            cursor_conn = get_connection()
            cursor = cursor_conn.cursor()
            cursor.execute("SELECT filename FROM documents WHERE document_id = ?", (chunk["document_id"],))
            doc = cursor.fetchone()
            cursor_conn.close()
            
            if doc and doc["filename"].lower() == expected_file.lower():
                rank_found = rank + 1  # 1-indexed rank
                break
                
        # Calculate MRR and Recall
        if rank_found != -1:
            rr = 1.0 / rank_found
            reciprocal_ranks.append(rr)
            if rank_found <= 1:
                recall_at_1 += 1
            if rank_found <= 3:
                recall_at_3 += 1
            if rank_found <= 5:
                recall_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)
            
        print(f"Query {idx+1}: '{query[:45]}...'")
        print(f"  Target Doc: {expected_file}")
        print(f"  Matched Rank: {rank_found if rank_found != -1 else 'NOT FOUND'} (Latency: {latency:.1f}ms)")
        print("-" * 70)
        
    # Summarize results
    mrr = sum(reciprocal_ranks) / total_queries
    r_at_1 = recall_at_1 / total_queries
    r_at_3 = recall_at_3 / total_queries
    r_at_5 = recall_at_5 / total_queries
    avg_latency = sum(latencies) / total_queries
    
    print("\n" + "=" * 70)
    print("                     EVALUATION REPORT SUMMARY                       ")
    print("=" * 70)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Average Search Latency  : {avg_latency:.2f} ms")
    print(f"Recall@1 Accuracy       : {r_at_1 * 100:.1f}%")
    print(f"Recall@3 Accuracy       : {r_at_3 * 100:.1f}%")
    print(f"Recall@5 Accuracy       : {r_at_5 * 100:.1f}%")
    print(f"Mean Reciprocal Rank    : {mrr:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
