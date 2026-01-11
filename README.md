This project investigates hallucination in multi-document retrieval augmented reasoning systems by designing an RAG based LLM that:
1. Retrieves evidence from multiple related technical documents
2. Performs multi-step reasoning over retrieved context
3. Estimates a confidence score for answers based on the reasoning
4. Explicitly refuses to answer when confidence score is low, hence the name. 
5. This project also analyses how retrieval quality, document overlap, and conflicting evidence impact hallucination behaviour in RAG systems.

The system is evaluated using custom benchmarks measuring accuracy, calibration, hallucination rate and refusal correctness. 


Research questions:
1. How does retrieval quality affect hallucination?
2. When should a system refuse to answer?
3. Can confidence calibration reduce overconfident errors? 