import os
import json
import urllib.request
import urllib.error
import re

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "mock")

def clean_json_response(text):
    """Strips any leading/trailing chat explanation text outside the first { and last }."""
    text_strip = text.strip()
    # Remove markdown code block fences if present
    text_strip = re.sub(r'^```(?:json)?', '', text_strip, flags=re.IGNORECASE)
    text_strip = re.sub(r'```$', '', text_strip).strip()
    
    start = text_strip.find('{')
    end = text_strip.rfind('}')
    if start != -1 and end != -1:
        return text_strip[start:end+1]
    return text_strip

class LocalTransformersLLM:
    """Runs a lightweight Qwen 0.5B model locally in Python using Hugging Face transformers."""
    _model = None
    _tokenizer = None

    @classmethod
    def get_model_and_tokenizer(cls):
        if cls._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"\n[Local Fallback] Loading native Python LLM: {model_id}...")
            print("[Local Fallback] This downloads ~950MB of model weights on its first run and runs entirely on CPU.")
            
            cls._tokenizer = AutoTokenizer.from_pretrained(model_id)
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32  # CPU friendly
            )
            print("[Local Fallback] Native Python LLM loaded successfully.")
        return cls._model, cls._tokenizer

    @classmethod
    def generate(cls, system_prompt, user_prompt):
        import torch
        model, tokenizer = cls.get_model_and_tokenizer()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        print("[Local Fallback] Running text generation on CPU...")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,  # Keep it highly deterministic
                do_sample=False
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class LLMClient:
    """Interfaces with a local OpenAI-compatible API to generate cited answers."""

    @classmethod
    def generate_answer(cls, query, retrieved_chunks):
        """
        Sends query and formatted contexts to the local LLM.
        Enforces a structured JSON output with citation tags mapping to retrieved chunks.
        """
        # Format the contexts
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            doc_name = chunk["document_name"]
            page = chunk["page_number"]
            sec = chunk["section_title"] or "N/A"
            ref = chunk["ref_number"] or "N/A"
            context_str += f"--- CONTEXT BLOCK [{idx + 1}] ---\n"
            context_str += f"Source: {doc_name} | Page: {page} | Section: {sec} | Ref: {ref}\n"
            context_str += f"Content: {chunk['chunk_text']}\n"
            context_str += "-------------------------\n\n"

        system_prompt = (
            "You are an expert Reserve Bank of India (RBI) compliance officer. "
            "Answer the query using ONLY the facts provided in the Context Blocks below. "
            "Do not assume, extrapolate, or bring in outside knowledge. "
            "Remember: The context may be silent on some aspects of the query. If the context does not explicitly mention "
            "the answer to the query, you MUST answer: 'The provided documents do not contain information regarding [insert query topic].' "
            "Do not speculate, extrapolate, or make up an answer.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output your answer STRICTLY as a JSON object with two fields: 'response' and 'citations'. Do not use any other keys.\n"
            "2. Write the answer inside 'response'. Every sentence in your response that makes factual claims from the context MUST end with an inline citation tag (e.g. [1], [2], or [3]). If you state that the information is not available, do NOT add any citation tags to that statement, and keep the 'citations' list empty.\n"
            "3. Only cite context blocks that are directly relevant to the query. Do not force multiple citations if the context does not support them.\n"
            "4. Keep factual details (limits, percentages, dates, and amounts) exactly as they are written in the context blocks to ensure accurate verification.\n"
            "5. For each citation tag used, include a record in the 'citations' list specifying:\n"
            "   - 'citation_tag': The tag used (e.g., '[1]')\n"
            "   - 'source_statement': The exact fact or claim in the response that uses this source\n"
            "   - 'source_block_index': The integer index (1-indexed) of the corresponding Context Block.\n\n"
            "JSON Format Example (When answer is found):\n"
            "{\n"
            "  \"response\": \"Regional Rural Banks have a target of 75 percent for priority sector lending [1]. Within this target, agriculture loans must constitute at least 18 percent of advances [2]. Small and marginal farmers are allocated a sub-target of 10 percent [3].\",\n"
            "  \"citations\": [\n"
            "    { \"citation_tag\": \"[1]\", \"source_statement\": \"Regional Rural Banks have a target of 75 percent for priority sector lending\", \"source_block_index\": 1 },\n"
            "    { \"citation_tag\": \"[2]\", \"source_statement\": \"agriculture loans must constitute at least 18 percent of advances\", \"source_block_index\": 2 },\n"
            "    { \"citation_tag\": \"[3]\", \"source_statement\": \"Small and marginal farmers are allocated a sub-target of 10 percent\", \"source_block_index\": 3 }\n"
            "  ]\n"
            "}\n\n"
            "JSON Format Example (When answer is NOT found):\n"
            "{\n"
            "  \"response\": \"The provided documents do not contain information regarding whether a customer can use PM-KISAN money as proof of income.\",\n"
            "  \"citations\": []\n"
            "}"
        )

        user_prompt = (
            f"Context Blocks:\n{context_str}\n\n"
            f"Query: {query}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Output your answer STRICTLY as a JSON object with two fields: 'response' and 'citations'. Do not use any other keys.\n"
            f"2. Write the answer inside 'response'. Every sentence in your response that makes factual claims from the context MUST end with an inline citation tag (e.g. [1], [2], or [3]). If you state that the information is not available, do NOT add any citation tags to that statement, and keep the 'citations' list empty.\n"
            f"3. Only cite context blocks that are directly relevant to the query. Do not force multiple citations if the context does not support them.\n"
            f"4. Keep factual details (limits, percentages, dates, and amounts) exactly as they are written in the context blocks to ensure accurate verification.\n"
            f"5. For each citation tag used, include a record in the 'citations' list specifying:\n"
            f"   - 'citation_tag': The tag used (e.g., '[1]')\n"
            f"   - 'source_statement': The exact fact or claim in the response that uses this source\n"
            f"   - 'source_block_index': The integer index (1-indexed) of the corresponding Context Block.\n\n"
            f"JSON Format Example (When answer is found):\n"
            f"{{\n"
            f"  \"response\": \"Regional Rural Banks have a target of 75 percent for priority sector lending [1]. Within this target, agriculture loans must constitute at least 18 percent of advances [2]. Small and marginal farmers are allocated a sub-target of 10 percent [3].\",\n"
            f"  \"citations\": [\n"
            f"    {{ \"citation_tag\": \"[1]\", \"source_statement\": \"Regional Rural Banks have a target of 75 percent for priority sector lending\", \"source_block_index\": 1 }},\n"
            f"    {{ \"citation_tag\": \"[2]\", \"source_statement\": \"agriculture loans must constitute at least 18 percent of advances\", \"source_block_index\": 2 }},\n"
            f"    {{ \"citation_tag\": \"[3]\", \"source_statement\": \"Small and marginal farmers are allocated a sub-target of 10 percent\", \"source_block_index\": 3 }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"JSON Format Example (When answer is NOT found):\n"
            f"{{\n"
            f"  \"response\": \"The provided documents do not contain information regarding whether a customer can use PM-KISAN money as proof of income.\",\n"
            f"  \"citations\": []\n"
            f"}}"
        )

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},  # Force JSON mode
            "temperature": 0.0
        }

        # Attempt REST API call to local server
        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(LLM_API_URL, data=req_data, headers=headers, method="POST")
        
        try:
            print(f"Connecting to local LLM server at {LLM_API_URL}...")
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = response.read().decode("utf-8")
                res_json = json.loads(res_data)
                content = res_json["choices"][0]["message"]["content"]
                cleaned_content = clean_json_response(content)
                return json.loads(cleaned_content)
        except (urllib.error.URLError, ConnectionRefusedError) as e:
            print(f"\nWARNING: Could not connect to local LLM server: {e}")
            print("[Local Fallback] Falling back to native in-memory Transformers execution...")
            try:
                raw_response = LocalTransformersLLM.generate(system_prompt, user_prompt)
                cleaned = clean_json_response(raw_response)
                return json.loads(cleaned)
            except Exception as inner_e:
                print(f"[Local Fallback] ERROR running native model: {inner_e}")
                return cls._get_mock_response(query, retrieved_chunks)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"response": f"Error generating answer: {e}", "citations": []}

    @staticmethod
    def _get_mock_response(query, chunks):
        """Generates a structured mock response for offline/testing fallback."""
        if not chunks:
            return {
                "response": "No relevant RBI documents were found to answer this question.",
                "citations": []
            }
            
        doc_name = chunks[0]["document_name"]
        page = chunks[0]["page_number"]
        sec = chunks[0]["section_title"] or "N/A"
        
        return {
            "response": f"[MOCK ANSWER] Based on {doc_name}, Page {page}, Section {sec}, the regulation states that: {chunks[0]['chunk_text'][:200]}... [1]",
            "citations": [
                {
                    "citation_tag": "[1]",
                    "source_statement": f"Based on {doc_name}, the regulation states that: {chunks[0]['chunk_text'][:100]}",
                    "source_block_index": 1
                }
            ]
        }

if __name__ == "__main__":
    # Quick client test
    mock_chunks = [{
        "chunk_id": "test-id",
        "document_name": "Master Circular - Kisan Credit Card (KCC) Scheme",
        "page_number": 4,
        "section_title": "5.1",
        "ref_number": "RBI/2017-18/4",
        "chunk_text": "The Kisan Credit Card (KCC) scheme aims at providing adequate and timely credit support from the banking system under a single window."
    }]
    ans = LLMClient.generate_answer("What is the aim of KCC?", mock_chunks)
    print("\n--- Generated Answer JSON ---")
    print(json.dumps(ans, indent=2))
