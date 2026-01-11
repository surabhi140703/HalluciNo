This project investigates hallucination in multi-document retrieval augmented reasoning systems by designing an RAG based LLM that:
1. Retrieves evidence from multiple technical documents
2. Performs multi-step reasoning over retrieved context
3. Estimates a confidence score for answers based on the reasoning
4. Explicitly refuses to answer when confidence score is low, hence the name. 
The system is evaluated using custom benchmarks measuring accuracy, calibration, hallucination rate and refusal correctness. 