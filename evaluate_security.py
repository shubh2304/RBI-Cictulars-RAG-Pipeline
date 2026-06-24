import os
import re
import sys

# Paths to evaluate
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(BASE_DIR, "main.py")
MODELS_PY = os.path.join(BASE_DIR, "database", "models.py")
LLM_CLIENT_PY = os.path.join(BASE_DIR, "generation", "llm_client.py")

class SecurityEvaluator:
    def __init__(self):
        self.findings = []
        self.score = 100

    def add_finding(self, control, severity, description):
        self.findings.append({
            "control": control,
            "severity": severity,
            "description": description
        })
        # Deduct score based on severity
        deduction = {"CRITICAL": 25, "HIGH": 15, "MEDIUM": 8, "LOW": 3}
        self.score = max(0, self.score - deduction.get(severity, 5))

    def evaluate_secrets(self):
        """A.8.9 Configuration Management: Check for hardcoded secrets."""
        # Simple heuristic regex for hardcoded secrets
        secret_patterns = [
            re.compile(r'(?:api_key|password|secret|auth_token)\s*=\s*["\'](?!http|localhost)[a-zA-Z0-9_\-]{8,}["\']', re.IGNORECASE),
        ]
        
        for file_path in [MAIN_PY, MODELS_PY, LLM_CLIENT_PY]:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            for pattern in secret_patterns:
                matches = pattern.findall(content)
                if matches:
                    for match in matches:
                        self.add_finding(
                            "A.8.9 Secrets Management",
                            "CRITICAL",
                            f"Potential hardcoded secret found in {os.path.basename(file_path)}: '{match.strip()}'"
                        )

    def evaluate_authentication(self):
        """A.8.2 Access Control & A.8.3 Securing Application Development: Check for endpoint auth."""
        if not os.path.exists(MAIN_PY):
            return
            
        with open(MAIN_PY, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if APIKeyHeader or security schema dependencies are imported/used
        has_auth_import = "APIKeyHeader" in content or "HTTPBearer" in content or "security" in content
        has_depends = "Depends(" in content
        
        # Check endpoints: /query and /ingest
        # Let's search if they are protected
        query_route_pattern = re.compile(r'@app\.post\("/query"\)\s*def\s+\w+\((.*?)\):', re.DOTALL)
        ingest_route_pattern = re.compile(r'@app\.post\("/ingest"\)\s*def\s+\w+\((.*?)\):', re.DOTALL)
        
        query_match = query_route_pattern.search(content)
        ingest_match = ingest_route_pattern.search(content)
        
        if query_match:
            params = query_match.group(1)
            if "Depends" not in params:
                self.add_finding(
                    "A.8.2 Access Control",
                    "HIGH",
                    "The `/query` endpoint does not appear to use a dependency injection for authentication (Depends)."
                )
                
        if ingest_match:
            params = ingest_match.group(1)
            if "Depends" not in params:
                self.add_finding(
                    "A.8.2 Access Control",
                    "CRITICAL",
                    "The administrative `/ingest` endpoint is public and does not require authentication."
                )

    def evaluate_input_validation(self):
        """A.8.3 Securing Application Development: Check for input validation and size limits."""
        if not os.path.exists(MAIN_PY):
            return
            
        with open(MAIN_PY, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if there is length validation on queries (e.g. max_length or len() check)
        # Check for query length check
        has_length_check = "len(query)" in content or "max_length" in content or "len(request.query)" in content
        if not has_length_check:
            self.add_finding(
                "A.8.3 Secure Coding (DoS)",
                "MEDIUM",
                "No query length limits detected. Large inputs may cause Denial of Service (DoS) by overloading embedding models."
            )

    def evaluate_error_leakage(self):
        """A.8.12 Data Leakage Prevention: Ensure detailed stack trace / error string is not sent to client."""
        if not os.path.exists(MAIN_PY):
            return
            
        with open(MAIN_PY, "r", encoding="utf-8") as f:
            content = f.read()

        # Search for exception message exposure (like raise HTTPException(..., detail=str(e)))
        error_leak_pattern = re.compile(r'raise\s+HTTPException\(.*detail\s*=\s*str\(e\)', re.IGNORECASE)
        matches = error_leak_pattern.findall(content)
        if matches:
            self.add_finding(
                "A.8.12 Information Leakage",
                "HIGH",
                f"Exposing raw exception message to client via `str(e)` in endpoint: {matches}"
            )

    def evaluate_sql_injection(self):
        """A.8.3 Secure Coding: Check for string interpolation in SQL executions."""
        sql_patterns = [
            re.compile(r'\.execute\(\s*f["\'].*?\{.*?\}["\']', re.IGNORECASE), # f-string interpolation
            re.compile(r'\.execute\(\s*["\'].*?%s.*?["\']\s*%', re.IGNORECASE), # percent formatting
            re.compile(r'\.execute\(\s*["\'].*?\{\}.*?["\']\.format', re.IGNORECASE), # format() string interpolation
        ]
        
        for file_path in [MAIN_PY, MODELS_PY, LLM_CLIENT_PY]:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            for pattern in sql_patterns:
                matches = pattern.findall(content)
                if matches:
                    for match in matches:
                        self.add_finding(
                            "A.8.3 Secure Coding (SQL Injection)",
                            "CRITICAL",
                            f"SQL Injection vulnerability in {os.path.basename(file_path)}: '{match.strip()}'"
                        )

    def evaluate_security_headers(self):
        """A.8.20 Network Security & A.8.24 Cryptography: Ensure CORS and secure headers are configured."""
        if not os.path.exists(MAIN_PY):
            return
            
        with open(MAIN_PY, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for CORSMiddleware
        has_cors = "CORSMiddleware" in content
        if not has_cors:
            self.add_finding(
                "A.8.20 Network Security",
                "MEDIUM",
                "CORS middleware is not configured, which may prevent domain origin restrictions."
            )
            
        # Check for secure header middleware
        has_headers = "X-Content-Type-Options" in content or "Content-Security-Policy" in content or "X-Frame-Options" in content
        if not has_headers:
            self.add_finding(
                "A.8.20 Network Security",
                "MEDIUM",
                "Secure HTTP response headers (e.g. X-Content-Type-Options, X-Frame-Options) are not enforced."
            )

    def evaluate_audit_logging(self):
        """A.8.15 Logging & Monitoring: Verify client identifier tracking in query logs."""
        if not os.path.exists(MODELS_PY):
            return
            
        with open(MODELS_PY, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if user_identity or user_id or client_identity is in query_logs table
        has_user_identity = "user_identity" in content.lower() or "user_id" in content.lower() or "api_key" in content.lower()
        if not has_user_identity:
            self.add_finding(
                "A.8.15 Audit Logging",
                "HIGH",
                "The query logs database schema does not store user or client identity information, preventing query attribution."
            )

    def run(self):
        print("=" * 80)
        print("              ISO/IEC 27001:2022 APPLICATION SECURITY REPORT            ")
        print("=" * 80)
        
        self.evaluate_secrets()
        self.evaluate_authentication()
        self.evaluate_input_validation()
        self.evaluate_error_leakage()
        self.evaluate_sql_injection()
        self.evaluate_security_headers()
        self.evaluate_audit_logging()
        
        if not self.findings:
            print("\n  [PASSED] Codebase conforms with all analyzed ISO 27001 controls.")
        else:
            print(f"\n  [FAILED] Found {len(self.findings)} security non-compliance issues:\n")
            for idx, f in enumerate(self.findings):
                print(f"  {idx+1}. [{f['severity']}] {f['control']}")
                print(f"     Description: {f['description']}")
                print("-" * 70)
                
        print("\n" + "=" * 80)
        print(f"  SECURITY COMPLIANCE SCORE: {self.score}/100")
        print("=" * 80)
        
        return len(self.findings) == 0

if __name__ == "__main__":
    evaluator = SecurityEvaluator()
    success = evaluator.run()
    if not success:
        sys.exit(1)
    sys.exit(0)
